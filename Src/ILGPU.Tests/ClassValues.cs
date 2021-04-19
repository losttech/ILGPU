using System;
using Xunit;
using Xunit.Abstractions;

namespace ILGPU.Tests
{
    public abstract class ClassValues : TestBase
    {
        protected ClassValues(
            ITestOutputHelper output,
            TestContext testContext)
            : base(output, testContext)
        { }

        internal class MathImpl
        {
#pragma warning disable CA1822 // Mark members as static
            public int Inc(int v) => v + 1;
#pragma warning restore CA1822 // Mark members as static
        }

        internal static void ThisIndependentMethodKernel(
            Index1 index, ArrayView<int> data)
        {
            data[index] = default(MathImpl).Inc(1);
        }

        [Fact]
        [KernelMethod(nameof(ThisIndependentMethodKernel))]
        public void ThisIndependentMethod()
        {
            using var output = Accelerator.Allocate<int>(1);
            Execute(1, output.View);

            var expected = new int[] { 2 };
            Verify(output, expected);
        }

        internal static void ThisIndependentLambdaKernel(
            Index1 index, ArrayView<int> data, Func<int, int> op)
        {
            data[index] = op(1);
        }

        static int Inc(int v) => v + 1;
        [Fact]
        public void ThisIndependentLambda()
        {
            Action<Index1, ArrayView<int>> kernel =
                (i, v) => ThisIndependentLambdaKernel(i, v, Inc);

            using var output = Accelerator.Allocate<int>(1);
            Execute(kernel.Method, new Index1(1), output.View);

            var expected = new int[] { 2 };
            Verify(output, expected);
        }
    }
}

#pragma warning restore xUnit1026 // Theory methods should use all of their parameters
