using System.Text;
using static RegressionTests.TestContext;

namespace RegressionTests;

internal static class Acuda64BitPointerBdaTests
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
        Check("64-bit acuda_compile_host found", host != null, host ?? "<not found>");
        if (host == null)
            return;

        string dir = Path.Combine(Path.GetTempPath(), "rtxdisam-acuda64-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            // 64-bit address space: .u64 pointer param + scalar value.
            string ptx64 = string.Join("\n", new[]
            {
                ".version 8.0",
                ".target sm_70",
                ".address_size 64",
                ".visible .entry k64(.param .u64 p0, .param .u32 n) {",
                "  .reg .u64 %rd<2>;",
                "  .reg .u32 %r<2>;",
                "  ld.param.u64 %rd1, [p0];",
                "  ld.param.u32 %r1, [n];",
                "  st.global.u32 [%rd1], %r1;",
                "  ret;",
                "}",
            });
            string ptx64Path = Path.Combine(dir, "k64.ptx");
            string out64 = Path.Combine(dir, "k64.akb");
            File.WriteAllText(ptx64Path, ptx64);
            var (ec64, out64Text) = RunAcudaCompileHost(host, ptx64Path, out64, "k64");
            Check("64-bit u64 pointer param compile", ec64 == 0 && File.Exists(out64), out64Text);
            Check("64-bit u64 pointer param classification",
                out64Text.Contains("ACuda params k64: dense=2 nptr=1 npush=1", StringComparison.Ordinal) &&
                out64Text.Contains("[0] kind=1", StringComparison.Ordinal) &&
                out64Text.Contains("[1] kind=0", StringComparison.Ordinal),
                out64Text);
        }
        finally
        {
            try { Directory.Delete(dir, true); } catch { }
        }
    }
}
