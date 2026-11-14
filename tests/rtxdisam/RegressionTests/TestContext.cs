using System.Text.Json;
using nvDismDecode;
using rtxdisam;

namespace RegressionTests;

internal static class TestContext
{
    public static int Passed;
    public static int Failed;

    public static void Init()
    {
        JsonOutput.Configure((value, indented) => JsonSerializer.Serialize(
            value, new JsonSerializerOptions { WriteIndented = indented }));
    }

    public static void Check(string name, bool ok, string? detail = null)
    {
        if (ok)
        {
            Passed++;
            return;
        }
        Failed++;
        Console.WriteLine($"FAIL {name}");
        if (detail != null)
            Console.WriteLine($"     {detail}");
    }

    public static void Equal(string name, string expected, string actual) =>
        Check(name, expected == actual, $"expected: {expected}\n     actual:   {actual}");
}
