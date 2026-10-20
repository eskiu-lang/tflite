#!/usr/bin/env python3
"""Generate src/raw.esk (and the layout check programs) from the TensorFlow Lite C headers.

Usage: tools/gen.py [INCLUDE_DIR]
INCLUDE_DIR is the root of a TensorFlow source tree, or of the headers vendored in
tools/include (the default), which come from TensorFlow's v2.21.0 tag.

The C API is made of ordinary exported functions, so each one becomes an Eskiu
extern with the C name. The functions of c_api.h, operator.h and xnnpack_delegate.h
are in libtensorflowlite_c; the GPU, Metal and Core ML delegates live in separate
libraries, so only their option structs and enums are generated here (<tflite> loads
their functions at run time).
"""
import os
import re
import sys

VERSION = "2.21.0"

KEYWORDS = set("""let int float double bool char string void struct packed union interface enum fn
operator asm volatile static escaping must_use async await const sizeof free_closure thread_create
thread_join for in while do if else switch match case default break return import extern intrinsic
alloc_with null true false try catch finally throw defer errdefer continue int8 int16 int32 int64 uint
uint8 uint16 uint32 uint64 type deprecated dbg""".split())

SCALARS = {
    "uint8_t": "uint8", "uint16_t": "uint16", "uint32_t": "uint32", "uint64_t": "uint64",
    "int8_t": "int8", "int16_t": "int16", "int32_t": "int", "int64_t": "int64",
    "size_t": "uint64", "int": "int", "float": "float", "double": "double", "bool": "bool",
    "char": "int8", "unsigned": "uint32",
}

# (header, what to take: "all" = types and functions, which are externs of
# libtensorflowlite_c; "types" = enums and structs; "enums" = enums, with its structs
# opaque)
HEADERS = [
    ("tensorflow/compiler/mlir/lite/core/c/tflite_types.h", "all"),
    ("tensorflow/lite/core/c/c_api_types.h", "all"),
    ("tensorflow/lite/core/async/c/types.h", "all"),
    ("tensorflow/lite/builtin_ops.h", "all"),
    ("tensorflow/lite/core/c/operator.h", "all"),
    ("tensorflow/lite/core/c/common.h", "enums"),
    ("tensorflow/lite/core/c/c_api.h", "all"),
    ("tensorflow/lite/core/c/c_api_opaque.h", "all"),
    ("tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h", "all"),
    ("tensorflow/lite/delegates/gpu/delegate_options.h", "types"),
    ("tensorflow/lite/delegates/gpu/metal_delegate.h", "types"),
    ("tensorflow/lite/delegates/coreml/coreml_delegate.h", "types"),
]

# Declared without TFL_CAPI_EXPORT and taking a TfLiteContext, which is not modelled.
SKIP_FUNCS = {"TfLiteXNNPackDelegateCreateWithThreadpool"}

APPLE_ONLY = {"metal_delegate.h"}

# Declared in c_api_opaque.h but not exported by the Windows DLL (MSVC exports only
# what TFL_CAPI_EXPORT marks, and the Windows build leaves these out).
NOT_ON_WINDOWS = {
    "TfLiteOpaqueDelegateCreate",
    "TfLiteOpaqueDelegateDelete",
    "TfLiteOpaqueDelegateGetData",
    "TfLiteOpaqueTensorBuilderCreate",
    "TfLiteOpaqueTensorBuilderDelete",
    "TfLiteOpaqueTensorBuilderSetAllocationType",
    "TfLiteOpaqueTensorBuilderSetData",
    "TfLiteOpaqueTensorBuilderSetQuantization",
    "TfLiteOpaqueTensorBuilderSetQuantizationParams",
    "TfLiteOpaqueTensorBuilderSetType",
    "TfLiteOpaqueTensorGetString",
    "TfLiteOpaqueTensorGetStringCount",
    "TfLiteOpaqueTensorSetAllocationTypeToDynamic",
    "TfLiteOpaqueTensorSetNonCpuAllocation",
    "TfLiteOpaqueTensorWriteString",
    "TfLiteOpaqueTensorWriteStrings"
}


def strip_comments(s):
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.S)
    return re.sub(r"//[^\n]*", "", s)


def preprocess(s):
    """Keeps what a C compiler sees with no macros defined: the #else branch of every
    #if/#ifdef (an #ifndef is taken), and no directive lines."""
    out, stack = [], []
    s = s.replace("\\\n", " ")
    for line in s.split("\n"):
        d = line.strip()
        if d.startswith("#"):
            words = d[1:].split()
            kw = words[0] if words else ""
            if kw in ("if", "ifdef"):
                stack.append([False, False])
            elif kw == "ifndef":
                stack.append([True, True])
            elif kw == "elif":
                stack[-1][0] = False
            elif kw == "else":
                stack[-1][0] = not stack[-1][1]
                stack[-1][1] = True
            elif kw == "endif":
                stack.pop()
            continue
        if all(on for on, _ in stack):
            out.append(line)
    return "\n".join(out)


WIDE = {"SIZE_MAX": "0xffffffffffffffff", "UINT64_MAX": "0xffffffffffffffff"}


def wide(members):
    return any(v in WIDE for _, v in members)


def ident(name):
    return name + "_" if name in KEYWORDS else name


def split_top(s, sep):
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch in "({":
            depth += 1
        elif ch in ")}":
            depth -= 1
        if ch == sep and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur)
    return out


def body_at(s, start):
    depth, i = 1, start
    while depth:
        depth += {"{": 1, "}": -1}.get(s[i], 0)
        i += 1
    return s[start:i - 1], i


class Gen:
    def __init__(self):
        self.opaque = []
        self.enums = []
        self.structs = []
        self.funcs = []
        self.typedefs = {}

    def known(self, name):
        return (name in self.opaque or name in [e[0] for e in self.enums]
                or name in [s[0] for s in self.structs])

    def ctype(self, t, where):
        """Maps a C type to Eskiu."""
        t = " ".join(t.replace("struct ", " ").replace("enum ", " ").split())
        stars = t.count("*")
        words = [w for w in t.replace("*", " ").split() if w not in ("const", "volatile", "TFL_CAPI_EXPORT", "extern")]
        if words[:2] == ["unsigned", "char"]:
            words = ["uint8_t"]
        elif words[:2] == ["unsigned", "int"]:
            words = ["unsigned"]
        base = words[0] if words else "void"
        base = self.typedefs.get(base, base)
        if base == "char" and stars >= 1:
            return "*" * (stars - 1) + "string"
        if base == "void":
            return "*" * stars + "void" if stars else "void"
        if base == "va_list" and stars == 0:
            return "va_list"
        if base in SCALARS:
            return "*" * stars + SCALARS[base]
        if self.known(base):
            return "*" * stars + base
        raise SystemExit("%s: unknown C type '%s'" % (where, t))

    def parse(self, text, externs, header):
        s = preprocess(strip_comments(text))
        s = s.replace('extern "C" {', "")
        # Named and anonymous typedef'd enums and structs.
        pos = 0
        rest = []
        for m in re.finditer(r"(typedef\s+)?(enum|struct)\s*(\w*)\s*\{", s):
            if m.start() < pos:
                continue
            body, end = body_at(s, m.end())
            tail = re.match(r"\s*(\w*)\s*;", s[end:])
            name = (tail.group(1) if tail and tail.group(1) else "") or m.group(3)
            rest.append(s[pos:m.start()])
            pos = end + (tail.end() if tail else 0)
            if m.group(2) == "enum":
                members = []
                for item in body.split(","):
                    item = " ".join(item.split())
                    if item:
                        k, _, v = item.partition("=")
                        members.append((k.strip(), v.strip()))
                self.enums.append((name, members, header))
            elif externs == "enums":
                if name and not self.known(name):
                    self.opaque.append(name)
            else:
                self.structs.append((name, body, header))
            if m.group(3) and m.group(3) != name:
                self.typedefs[m.group(3)] = name
        rest.append(s[pos:])
        s = "".join(rest)
        for m in re.finditer(r"typedef\s+struct\s+(\w+)\s+(\w+)\s*;", s):
            if m.group(1) == m.group(2) and not self.known(m.group(1)):
                self.opaque.append(m.group(1))
        for m in re.finditer(r"(?<!typedef )\bstruct\s+(\w+)\s*;", s):
            if not self.known(m.group(1)):
                self.opaque.append(m.group(1))
        for m in re.finditer(r"typedef\s+(\w+)\s+(\w+)\s*;", s):
            self.typedefs[m.group(2)] = m.group(1)
        if externs != "all":
            return
        for decl in split_top(s, ";"):
            decl = " ".join(decl.split())
            if "typedef" in decl or "(" not in decl or decl.startswith("struct "):
                continue
            m = re.match(r"(.*?)\b(\w+)\s*\((.*)\)$", decl)
            if not m:
                raise SystemExit("%s: cannot parse '%s'" % (header, decl))
            if m.group(2) in SKIP_FUNCS:
                continue
            self.funcs.append((m.group(1), m.group(2), m.group(3), header))

    def finish(self):
        self.struct_fields = []
        for name, body, header in self.structs:
            fields = []
            for decl in split_top(body, ";"):
                decl = " ".join(decl.split())
                if not decl:
                    continue
                fm = re.match(r"(.*?)(\w+)\s*(\[\s*(\w+)\s*\])?$", decl)
                t = self.ctype(fm.group(1), "%s.%s" % (name, fm.group(2)))
                if fm.group(3):
                    t = "%s[%s]" % (t, fm.group(4))
                fields.append((t, ident(fm.group(2)), fm.group(2)))
            self.struct_fields.append((name, fields, header))
        self.externs = []
        for ret, name, params, header in self.funcs:
            rt = self.ctype(ret, name)
            ps = []
            if params.strip() and params.strip() != "void":
                for i, p in enumerate(split_top(params, ",")):
                    p = " ".join(p.split())
                    if p == "...":
                        ps.append(("...", ""))
                        continue
                    if re.match(r".*\(\s*\*\s*\w+\s*\)\s*\(.*\)$", p):
                        ps.append(("*void", ident(re.match(r".*\(\s*\*\s*(\w+)", p).group(1))))
                        continue
                    pm = re.match(r"(.*?)\s*(\w+)\s*(\[\s*\w*\s*\])?$", p)
                    if not pm:
                        raise SystemExit("%s: cannot parse parameter '%s'" % (name, p))
                    ptype, pname = pm.group(1), pm.group(2)
                    if pm.group(3):
                        ptype += "*"
                    ps.append((self.ctype(ptype, "%s(%s)" % (name, p)), ident(pname)))
            self.externs.append((rt, name, ps, header))

    def emit(self):
        out = ["// Generated by tools/gen.py from the TensorFlow Lite %s C headers. Do not edit." % VERSION,
               "// size_t is uint64: the prebuilt libraries are for 64-bit platforms only.",
               "//",
               "// Every function of c_api.h, operator.h and xnnpack_delegate.h, with its C name.",
               "// The GPU (OpenCL/OpenGL), Metal and Core ML delegates are separate libraries:",
               "// their option structs and enums are here, and <tflite> loads their functions.",
               "", '#pragma link("tensorflowlite_c")', ""]
        out.append('const string TFLITE_HEADERS_VERSION = "%s";' % VERSION)
        out.append("")
        for name in self.opaque:
            out.append("struct %s { int _opaque; }" % name)
        out.append("")
        out.append("// TFLITE_WITH_STABLE_ABI is off in the prebuilt libraries.")
        out.append("type TfLiteOpaqueDelegate = TfLiteDelegate;")
        out.append("")
        for name, members, header in self.enums:
            if wide(members) or not name:
                # An anonymous enum, or one whose values do not fit an int, is constants.
                t = "uint64" if wide(members) else "int"
                if name:
                    out.append("type %s = uint64;" % name)
                for k, v in members:
                    out.append("const %s %s = %s;" % (t, k, WIDE.get(v, v)))
                out.append("")
                continue
            out.append("enum %s {" % name)
            out.append(",\n".join("    %s%s" % (k, " = " + v if v else "") for k, v in members))
            out.append("}")
            out.append("")
        for name, fields, header in self.struct_fields:
            out.append("struct %s {" % name)
            for t, f, _ in fields:
                out.append("    %s %s;" % (t, f))
            out.append("}")
            out.append("")
        last = None
        for rt, name, ps, header in self.externs:
            if header != last:
                out.append("// %s" % header)
                last = header
            line = "extern %s %s(%s);" % (rt, name, ", ".join(("%s %s" % p).strip() for p in ps))
            if name in NOT_ON_WINDOWS:
                out += ["#ifndef _WIN32", line, "#endif"]
            else:
                out.append(line)
        return "\n".join(out).rstrip() + "\n"

    def emit_layout(self):
        c = ["#include <stdio.h>", "#include <stddef.h>"]
        for h, _ in HEADERS:
            if os.path.basename(h) in APPLE_ONLY:
                c += ["#ifdef __APPLE__", '#include "%s"' % h, "#endif"]
            else:
                c.append('#include "%s"' % h)
        c.append("int main(void) {")
        e = ["import <io>;", "import <tflite/raw>;", "", "int main() {"]

        def guard(lines, header, on):
            if os.path.basename(header) in APPLE_ONLY:
                lines.append("#ifdef __APPLE__" if on else "#endif")

        for name, fields, header in self.struct_fields:
            guard(c, header, True)
            guard(e, header, True)
            c.append('    printf("%s %%zu\\n", sizeof(%s));' % (name, name))
            e.append('    printf("%s %%d\\n", (int)sizeof(%s));' % (name, name))
            e.append("    { %s v;" % name)
            for t, f, cf in fields:
                c.append('    printf("%s.%s %%zu\\n", offsetof(%s, %s));' % (name, cf, name, cf))
                e.append('      printf("%s.%s %%d\\n", (int)((*uint8)&v.%s - (*uint8)&v));' % (name, cf, f))
            e.append("    }")
            guard(c, header, False)
            guard(e, header, False)
        for name, members, header in self.enums:
            guard(c, header, True)
            guard(e, header, True)
            for k, _ in members:
                if wide(members):
                    c.append('    printf("%s %%llu\\n", (unsigned long long)%s);' % (k, k))
                    e.append('    printf("%s %%llu\\n", (uint64)%s);' % (k, k))
                else:
                    c.append('    printf("%s %%d\\n", (int)%s);' % (k, k))
                    e.append('    printf("%s %%d\\n", (int)%s);' % (k, k))
            guard(c, header, False)
            guard(e, header, False)
        # Every extern is linked: a function the library does not export fails here.
        n = len(self.externs)
        c += ["#ifdef _WIN32", '    printf("functions %%d\\n", %d);' % (n - len(NOT_ON_WINDOWS)),
              "#else", '    printf("functions %%d\\n", %d);' % n, "#endif"]
        e.append("    int linked = 0;")
        for rt, name, ps, header in self.externs:
            line = "    if ((*void)%s != null) linked++;" % name
            if name in NOT_ON_WINDOWS:
                e += ["#ifndef _WIN32", line, "#endif"]
            else:
                e.append(line)
        e.append('    printf("functions %d\\n", linked);')
        c += ["    return 0;", "}"]
        e += ["    return 0;", "}"]
        return "\n".join(c) + "\n", "\n".join(e) + "\n"


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    inc = sys.argv[1] if len(sys.argv) > 1 else os.path.join(root, "tools", "include")
    g = Gen()
    for h, externs in HEADERS:
        with open(os.path.join(inc, h)) as f:
            g.parse(f.read(), externs, h)
    g.finish()
    with open(os.path.join(root, "src", "raw.esk"), "w") as f:
        f.write(g.emit())
    c, e = g.emit_layout()
    os.makedirs(os.path.join(root, "tools", "layout"), exist_ok=True)
    with open(os.path.join(root, "tools", "layout", "layout.c"), "w") as f:
        f.write(c)
    with open(os.path.join(root, "tools", "layout", "layout.esk"), "w") as f:
        f.write(e)
    print("src/raw.esk: %d functions, %d structs, %d enums, %d opaque types" %
          (len(g.externs), len(g.struct_fields), len(g.enums), len(g.opaque)))


main()
