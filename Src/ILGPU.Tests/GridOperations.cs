using ILGPU.Runtime;
using System;
using System.Diagnostics;
using Xunit;
using Xunit.Abstractions;

namespace ILGPU.Tests
{
    [Collection("DimensionOperations")]
    public abstract class GridOperations : TestBase
    {
        protected GridOperations(ITestOutputHelper output, TestContext testContext)
            : base(output, testContext)
        { }

        internal static void GridDimensionKernel(
            ArrayView1D<int, Stride1D.Dense> data)
        {
            data[0] = Grid.DimX;
            data[1] = Grid.DimY;
            data[2] = Grid.DimZ;

            Debug.Assert(Grid.IdxX < Grid.DimX);
            Debug.Assert(Grid.IdxY < Grid.DimY);
            Debug.Assert(Grid.IdxZ < Grid.DimZ);
        }

        [Theory]
        [InlineData(1, 0, 0)]
        [InlineData(0, 1, 0)]
        [InlineData(0, 0, 1)]
        [KernelMethod(nameof(GridDimensionKernel))]
        public void GridDimension(int xMask, int yMask, int zMask)
        {
            for (int i = 2; i <= Accelerator.MaxNumThreadsPerGroup; i <<= 1)
            {
                using var buffer = Accelerator.Allocate1D<int>(3);
                var extent = new KernelConfig(
                    new Index3D(
                        Math.Max(i * xMask, 1),
                        Math.Max(i * yMask, 1),
                        Math.Max(i * zMask, 1)),
                    Index3D.One);

                Execute(extent, buffer.View);

                var expected = new int[]
                {
                    extent.GridDim.X,
                    extent.GridDim.Y,
                    extent.GridDim.Z,
                };
                Verify(buffer.View, expected);
            }
        }

        internal static void GridLaunchDimensionKernel(
            ArrayView1D<int, Stride1D.Dense> data)
        {
            data[0] = Grid.DimX;
        }

        // This test is one-dimensional and uses small sizes for the sake of passing
        // tests on the CI machine, but on a machine with more threads it works
        // for higher dimensions and higher sizes.
        [Fact]
        public void GridLaunchDimension()
        {
            using var buffer = Accelerator.Allocate1D<int>(1);
            var kernel = Accelerator.LoadStreamKernel<ArrayView1D<int, Stride1D.Dense>>
                (GridLaunchDimensionKernel);

            kernel((1, 2), buffer.View);
            Accelerator.Synchronize();

            var data = buffer.GetAs1DArray();
            int expected = 1;

            Assert.Equal(expected, data[0]);
        }
    }
}
