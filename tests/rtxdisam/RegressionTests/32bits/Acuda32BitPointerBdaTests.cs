using System.Text;
using static RegressionTests.TestContext;

namespace RegressionTests;

internal static class Acuda32BitPointerBdaTests
{
    public static void Run()
    {
        static string? FindAcudaCompileHost()
        {
            static string? Probe(string baseDir)
            {
                foreach (string rel in new[]
                {
                    Path.Combine("out", "win-x64", "rtxdiasm_ptx-to-spirv_compiler.exe"),
                    Path.Combine("out", "win-x64", "acuda_compile_host.exe"),
                    Path.Combine("rtxdiasm_modules", "rtxdiasm-ptx-to-spirv", "build", "Release", "rtxdiasm_ptx-to-spirv_compiler.exe"),
                    Path.Combine("rtxdiasm_modules", "rtxdiasm-ptx-to-spirv", "build", "rtxdiasm_ptx-to-spirv_compiler.exe"),
                    Path.Combine("src", "acuda", "bin", "release", "Release", "rtxdiasm_ptx-to-spirv_compiler.exe"),
                    Path.Combine("src", "acuda", "bin", "release", "rtxdiasm_ptx-to-spirv_compiler.exe"),
                    Path.Combine("src", "acuda", "bin", "release", "Release", "acuda_compile_host.exe"),
                    Path.Combine("src", "acuda", "bin", "release", "acuda_compile_host.exe"),
                    Path.Combine("src", "acuda", "bin", "release", "acuda_compile_host"),
                    Path.Combine("src", "acuda", "bin", "debug", "rtxdiasm_ptx-to-spirv_compiler.exe"),
                    Path.Combine("src", "acuda", "bin", "debug", "acuda_compile_host.exe"),
                    Path.Combine("src", "acuda", "bin", "debug", "acuda_compile_host"),
                    "rtxdiasm_ptx-to-spirv_compiler.exe",
                    "acuda_compile_host.exe",
                    "acuda_compile_host",
                })
                {
                    string p = Path.GetFullPath(Path.Combine(baseDir, rel));
                    if (File.Exists(p)) return p;
                }
                return null;
            }

            foreach (string start in new[]
            {
                Environment.CurrentDirectory,
                AppContext.BaseDirectory,
            })
            {
                string dir = Path.GetFullPath(start);
                for (int i = 0; i < 8; i++)
                {
                    string? p = Probe(dir);
                    if (p != null) return p;
                    string? parent = Path.GetDirectoryName(dir);
                    if (parent == null || parent == dir) break;
                    dir = parent;
                }
            }
            return null;
        }

        static (int ExitCode, string Output) RunAcudaCompileHost(
            string exe, string ptxPath, string outPath, string entry)
        {
            var psi = new System.Diagnostics.ProcessStartInfo(exe)
            {
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true,
            };
            psi.ArgumentList.Add(ptxPath);
            psi.ArgumentList.Add(outPath);
            psi.ArgumentList.Add(entry);
            psi.Environment["ACUDA_DUMP_PARAMS"] = "1";
            using var proc = System.Diagnostics.Process.Start(psi);
            if (proc == null) return (-1, "failed to start acuda_compile_host");
            string stdout = proc.StandardOutput.ReadToEnd();
            string stderr = proc.StandardError.ReadToEnd();
            proc.WaitForExit();
            return (proc.ExitCode, stdout + stderr);
        }

        string? host = FindAcudaCompileHost();
        Check("32-bit acuda_compile_host found", host != null, host ?? "<not found>");
        if (host == null)
            return;

        string dir = Path.Combine(Path.GetTempPath(), "rtxdisam-acuda32-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            // 32-bit address space: plain .u32 params used as addresses must
            // classify as pointer bindings (kind 1), value-only as scalar (kind 0).
            // Also exercises BDA physical pointer type emission in compile_host.
            string ptxPtr = string.Join("\n", new[]
            {
                ".version 4.1",
                ".target sm_52",
                ".address_size 32",
                ".visible .entry k_ptr(.param .u32 p0, .param .u32 p1) {",
                "  .reg .u32 %r<4>;",
                "  ld.param.u32 %r1, [p0];",
                "  ld.param.u32 %r2, [p1];",
                "  st.global.u32 [%r1], %r2;",
                "  ret;",
                "}",
            });
            string ptxPath = Path.Combine(dir, "k_ptr.ptx");
            string outPath = Path.Combine(dir, "k_ptr.akb");
            File.WriteAllText(ptxPath, ptxPtr);
            var (ec, output) = RunAcudaCompileHost(host, ptxPath, outPath, "k_ptr");
            Check("32-bit u32 pointer params compile", ec == 0 && File.Exists(outPath), output);
            Check("32-bit u32 pointer params classification",
                output.Contains("ACuda params k_ptr: dense=2 nptr=1 npush=1", StringComparison.Ordinal) &&
                output.Contains("[0] kind=1", StringComparison.Ordinal) &&
                output.Contains("[1] kind=0", StringComparison.Ordinal),
                output);

            // Mixed: multiple u32 pointer params + scalar count in 32-bit space.
            string ptxMix = string.Join("\n", new[]
            {
                ".version 4.1",
                ".target sm_52",
                ".address_size 32",
                ".visible .entry k_mix(.param .u32 a, .param .u32 b, .param .u32 c, .param .u32 d, .param .u32 n) {",
                "  .reg .u32 %r<8>;",
                "  ld.param.u32 %r1, [a];",
                "  ld.param.u32 %r2, [b];",
                "  ld.param.u32 %r3, [c];",
                "  ld.param.u32 %r4, [d];",
                "  ld.param.u32 %r5, [n];",
                "  ld.global.u32 %r6, [%r1];",
                "  add.u32 %r7, %r6, %r5;",
                "  st.global.u32 [%r2], %r7;",
                "  st.global.u32 [%r3], %r6;",
                "  st.global.u32 [%r4], %r5;",
                "  ret;",
                "}",
            });
            string mixPtx = Path.Combine(dir, "k_mix.ptx");
            string mixOut = Path.Combine(dir, "k_mix.akb");
            File.WriteAllText(mixPtx, ptxMix);
            var (ec2, out2) = RunAcudaCompileHost(host, mixPtx, mixOut, "k_mix");
            Check("32-bit mixed u32 ptr/scalar compile", ec2 == 0 && File.Exists(mixOut), out2);
            Check("32-bit mixed u32 ptr/scalar classification",
                out2.Contains("ACuda params k_mix: dense=5 nptr=4 npush=1", StringComparison.Ordinal) &&
                out2.Contains("[0] kind=1", StringComparison.Ordinal) &&
                out2.Contains("[1] kind=1", StringComparison.Ordinal) &&
                out2.Contains("[2] kind=1", StringComparison.Ordinal) &&
                out2.Contains("[3] kind=1", StringComparison.Ordinal) &&
                out2.Contains("[4] kind=0", StringComparison.Ordinal),
                out2);
        }
        finally
        {
            try { Directory.Delete(dir, true); } catch { }
        }
    }
}
