namespace RegressionTests;

internal static class Program
{
    public static int Main()
    {
        TestContext.Init();

        DecoderIsaTests.Run();
        FatbinDecodingTests.Run();
        HostContainerAndApiTests.Run();
        Acuda32BitPointerBdaTests.Run();
        HostContainerTests32.Run();
        FatbinPtx32BitTests.Run();
        Acuda64BitPointerBdaTests.Run();

        Console.WriteLine($"{TestContext.Passed} passed, {TestContext.Failed} failed");
        return TestContext.Failed == 0 ? 0 : 1;
    }
}
