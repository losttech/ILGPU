﻿// ---------------------------------------------------------------------------------------
//                                        ILGPU
//                        Copyright (c) 2016-2020 Marcel Koester
//                                    www.ilgpu.net
//
// File: Accelerator.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details
// ---------------------------------------------------------------------------------------

using ILGPU.Backends;
using ILGPU.Frontend.Intrinsic;
using ILGPU.Resources;
using ILGPU.Util;
using System;
using System.Collections.Immutable;
using System.Diagnostics;
using System.IO;
using System.Threading;

namespace ILGPU.Runtime
{
    /// <summary>
    /// Represents the general type of an accelerator.
    /// </summary>
    public enum AcceleratorType : int
    {
        /// <summary>
        /// Represents a CPU accelerator.
        /// </summary>
        CPU,

        /// <summary>
        /// Represents a Cuda accelerator.
        /// </summary>
        Cuda,

        /// <summary>
        /// Represents an OpenCL accelerator (CPU/GPU via OpenCL).
        /// </summary>
        OpenCL,
    }

    /// <summary>
    /// Represents an abstract accelerator extension that can store additional data.
    /// </summary>
    public abstract class AcceleratorExtension : CachedExtension { }

    /// <summary>
    /// Represents a general abstract accelerator.
    /// </summary>
    /// <remarks>Members of this class are not thread safe.</remarks>
    public abstract partial class Accelerator :
        CachedExtensionBase<AcceleratorExtension>
    {
        #region Static

        /// <summary>
        /// Detects all accelerators.
        /// </summary>
        static Accelerator()
        {
            var accelerators = ImmutableArray.CreateBuilder<AcceleratorId>(4);
            accelerators.AddRange(CPU.CPUAccelerator.CPUAccelerators);
            accelerators.AddRange(Cuda.CudaAccelerator.CudaAccelerators);
            accelerators.AddRange(OpenCL.CLAccelerator.CLAccelerators);
            Accelerators = accelerators.ToImmutable();
        }

        /// <summary>
        /// Represents all available accelerators.
        /// </summary>
        public static ImmutableArray<AcceleratorId> Accelerators { get; }

        /// <summary>
        /// Creates the specified accelerator using the provided accelerator id.
        /// </summary>
        /// <param name="context">The ILGPU context.</param>
        /// <param name="acceleratorId">The specified accelerator id.</param>
        /// <returns>The created accelerator.</returns>
        public static Accelerator Create(Context context, AcceleratorId acceleratorId)
        {
            if (context == null)
                throw new ArgumentNullException(nameof(context));

            return acceleratorId switch
            {
                CPU.CPUAcceleratorId _ => new CPU.CPUAccelerator(context),
                Cuda.CudaAcceleratorId cudaId =>
                    new Cuda.CudaAccelerator(context, cudaId.DeviceId),
                OpenCL.CLAcceleratorId clId =>
                    new OpenCL.CLAccelerator(context, clId),
                _ => throw new ArgumentException(
                    RuntimeErrorMessages.NotSupportedTargetAccelerator,
                    nameof(acceleratorId)),
            };
        }

        /// <summary>
        /// Returns the current accelerator type.
        /// </summary>
        /// <remarks>
        /// Note that this static property is also accessible within kernels.
        /// </remarks>
        public static AcceleratorType CurrentType
        {
            [AcceleratorIntrinsic(AcceleratorIntrinsicKind.CurrentType)]
            get => AcceleratorType.CPU;
        }

        #endregion

        #region Events

        /// <summary>
        /// Will be raised if the accelerator is disposed.
        /// </summary>
        public event EventHandler Disposed;

        #endregion

        #region Instance

        /// <summary>
        /// Main object for accelerator synchronization.
        /// </summary>
        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        private readonly object syncRoot = new object();

        /// <summary>
        /// The default memory cache for operations that require additional
        /// temporary memory.
        /// </summary>
        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        private readonly MemoryBufferCache memoryCache;

        /// <summary>
        /// The current volatile native pointer of this instance.
        /// </summary>
        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        private volatile IntPtr nativePtr;

        /// <summary>
        /// Constructs a new accelerator.
        /// </summary>
        /// <param name="context">The target context.</param>
        /// <param name="type">The target accelerator type.</param>
        internal Accelerator(Context context, AcceleratorType type)
        {
            Context = context ?? throw new ArgumentNullException(nameof(context));
            AcceleratorType = type;
            InstanceId = InstanceId.CreateNew();

            InitKernelCache();
            InitLaunchCache();
            InitGC();

            memoryCache = new MemoryBufferCache(this);
        }

        #endregion

        #region Properties

        /// <summary>
        /// Returns the internal unique accelerator instance id.
        /// </summary>
        internal InstanceId InstanceId { get; }

        /// <summary>
        /// Returns the associated ILGPU context.
        /// </summary>
        public Context Context { get; }

        /// <summary>
        /// Returns the default stream of this accelerator.
        /// </summary>
        public AcceleratorStream DefaultStream { get; protected set; }

        /// <summary>
        /// Returns the type of the accelerator.
        /// </summary>
        public AcceleratorType AcceleratorType { get; }

        /// <summary>
        /// Returns the current native accelerator pointer.
        /// </summary>
        public IntPtr NativePtr
        {
            get => nativePtr;
            protected set => nativePtr = value;
        }

        /// <summary>
        /// Returns the name of the device.
        /// </summary>
        public string Name { get; protected set; }

        /// <summary>
        /// Returns the memory size in bytes.
        /// </summary>
        public long MemorySize { get; protected set; }

        /// <summary>
        /// Returns the max grid size.
        /// </summary>
        public Index3D MaxGridSize { get; protected set; }

        /// <summary>
        /// Returns the max group size.
        /// </summary>
        public Index3D MaxGroupSize { get; protected set; }

        /// <summary>
        /// Returns the maximum number of threads in a group.
        /// </summary>
        public int MaxNumThreadsPerGroup { get; protected set; }

        /// <summary>
        /// Returns the maximum number of threads in a group.
        /// </summary>
        [Obsolete("Use MaxNumThreadsPerGroup instead")]
        public int MaxThreadsPerGroup => MaxNumThreadsPerGroup;

        /// <summary>
        /// Returns the maximum number of shared memory per thread group in bytes.
        /// </summary>
        public int MaxSharedMemoryPerGroup { get; protected set; }

        /// <summary>
        /// Returns the maximum number of constant memory in bytes.
        /// </summary>
        public int MaxConstantMemory { get; protected set; }

        /// <summary>
        /// Return the warp size.
        /// </summary>
        public int WarpSize { get; protected set; }

        /// <summary>
        /// Returns the number of available multiprocessors.
        /// </summary>
        public int NumMultiprocessors { get; protected set; }

        /// <summary>
        /// Returns the maximum number of threads per multiprocessor.
        /// </summary>
        public int MaxNumThreadsPerMultiprocessor { get; protected set; }

        /// <summary>
        /// Returns the maximum number of threads of this accelerator.
        /// </summary>
        public int MaxNumThreads => NumMultiprocessors * MaxNumThreadsPerMultiprocessor;

        /// <summary>
        /// Returns a kernel extent (a grouped index) with the maximum number of groups
        /// using the maximum number of threads per group to launch common grid-stride
        /// loop kernels.
        /// </summary>
        public (Index1D, Index1D) MaxNumGroupsExtent =>
            (NumMultiprocessors *
                (MaxNumThreadsPerMultiprocessor / MaxNumThreadsPerGroup),
            MaxNumThreadsPerGroup);

        /// <summary>
        /// Returns the primary backend of this accelerator.
        /// </summary>
        public Backend Backend { get; private set; }

        /// <summary>
        /// Returns the supported capabilities of this accelerator.
        /// </summary>
        public CapabilityContext Capabilities { get; protected set; }

        /// <summary>
        /// Returns the default memory-buffer cache that can be used by several
        /// operations.
        /// </summary>
        public MemoryBufferCache MemoryCache => memoryCache;

        #endregion

        #region Methods

        /// <summary>
        /// Initializes the current accelerator instance.
        /// </summary>
        /// <param name="backend">The backend to use.</param>
        protected void Init(Backend backend)
        {
            Backend = backend;
            Capabilities = backend.Capabilities;
            OnAcceleratorCreated();
        }

        /// <summary>
        /// Invoked when the accelerator instance has been created.
        /// </summary>
        protected void OnAcceleratorCreated() => Context.OnAcceleratorCreated(this);

        /// <summary>
        /// Creates a new accelerator extension using the given provider.
        /// </summary>
        /// <typeparam name="TExtension">The type of the extension to create.</typeparam>
        /// <typeparam name="TExtensionProvider">
        /// The extension provided type to create the extension.
        /// </typeparam>
        /// <param name="provider">
        /// The extension provided to create the extension.
        /// </param>
        /// <returns>The created extension.</returns>
        public abstract TExtension CreateExtension<TExtension, TExtensionProvider>(
            TExtensionProvider provider)
            where TExtensionProvider : IAcceleratorExtensionProvider<TExtension>;

        /// <summary>
        /// Creates a new accelerator stream.
        /// </summary>
        /// <returns>The created accelerator stream.</returns>
        public AcceleratorStream CreateStream()
        {
            Bind(); return CreateStreamInternal();
        }

        /// <summary>
        /// Creates a new accelerator stream.
        /// </summary>
        /// <returns>The created accelerator stream.</returns>
        protected abstract AcceleratorStream CreateStreamInternal();

        /// <summary>
        /// Synchronizes pending operations.
        /// </summary>
        public void Synchronize()
        {
            Bind(); SynchronizeInternal();
        }

        /// <summary>
        /// Synchronizes pending operations.
        /// </summary>
        protected abstract void SynchronizeInternal();

        /// <summary>
        /// Clears all internal caches.
        /// </summary>
        /// <param name="mode">The clear mode.</param>
        public override void ClearCache(ClearCacheMode mode)
        {
            lock (syncRoot)
            {
                Backend.ClearCache(mode);
                ClearKernelCache_SyncRoot();
                ClearLaunchCache_SyncRoot();
                base.ClearCache(mode);
            }
        }

        /// <summary>
        /// Prints device information to the standard <see cref="Console.Out"/> stream.
        /// </summary>
        public void PrintInformation() => PrintInformation(Console.Out);

        /// <summary>
        /// Prints device information to the given text writer.
        /// </summary>
        /// <param name="writer">The target text writer to write to.</param>
        public void PrintInformation(TextWriter writer)
        {
            if (writer is null)
                throw new ArgumentNullException(nameof(writer));

            PrintHeader(writer);
            PrintGeneralInfo(writer);
        }

        /// <summary>
        /// Prints general header information that should appear at the top.
        /// </summary>
        /// <param name="writer">The target text writer to write to.</param>
        protected virtual void PrintHeader(TextWriter writer)
        {
            writer.Write("Device: ");
            writer.Write(Name);
            writer.WriteLine(" [ILGPU InstanceId: {0}]", InstanceId);
        }

        /// <summary>
        /// Print general GPU specific information to the given text writer.
        /// </summary>
        /// <param name="writer">The target text writer to write to.</param>
        protected virtual void PrintGeneralInfo(TextWriter writer)
        {
            writer.Write("  Number of multiprocessors:               ");
            writer.WriteLine(NumMultiprocessors);

            writer.Write("  Max number of threads/multiprocessor:    ");
            writer.WriteLine(MaxNumThreadsPerMultiprocessor);

            writer.Write("  Max number of threads/group:             ");
            writer.WriteLine(MaxNumThreadsPerGroup);

            writer.Write("  Max number of total threads:             ");
            writer.WriteLine(MaxNumThreads);

            writer.Write("  Max dimension of a group size:           ");
            writer.WriteLine(MaxGroupSize.ToString());

            writer.Write("  Max dimension of a grid size:            ");
            writer.WriteLine(MaxGridSize.ToString());

            writer.Write("  Total amount of global memory:           ");
            writer.WriteLine(
                "{0} bytes, {1} MB",
                MemorySize,
                MemorySize / (1024 * 1024));

            writer.Write("  Total amount of constant memory:         ");
            writer.WriteLine(
                "{0} bytes, {1} KB",
                MaxConstantMemory,
                MaxConstantMemory / 1024);

            writer.Write("  Total amount of shared memory per group: ");
            writer.WriteLine(
                "{0} bytes, {1} KB",
                MaxSharedMemoryPerGroup,
                MaxSharedMemoryPerGroup / 1024);
        }

        #endregion

        #region Allocation

        /// <summary>
        /// Allocates a buffer with the specified size in bytes on this accelerator.
        /// </summary>
        /// <param name="length">The number of elements to allocate.</param>
        /// <param name="elementSize">The size of a single element in bytes.</param>
        /// <returns>An allocated buffer on the this accelerator.</returns>
        public MemoryBuffer AllocateRaw(long length, int elementSize)
        {
            if (length < 0)
                throw new ArgumentOutOfRangeException(nameof(length));
            if (elementSize < 1)
                throw new ArgumentOutOfRangeException(nameof(elementSize));

            Bind();
            return AllocateRawInternal(length, elementSize);
        }

        /// <summary>
        /// Allocates a buffer with the specified number of elements on this accelerator.
        /// </summary>
        /// <param name="length">The number of elements to allocate.</param>
        /// <param name="elementSize">The size of a single element in bytes.</param>
        /// <returns>An allocated buffer on the this accelerator.</returns>
        protected abstract MemoryBuffer AllocateRawInternal(
            long length,
            int elementSize);

        /// <summary>
        /// Allocates an n-D buffer with the specified number of elements on this
        /// accelerator times the total stride length.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <typeparam name="TExtent">The extent type of the buffer.</typeparam>
        /// <typeparam name="TStrideIndex">The buffer stride index type.</typeparam>
        /// <typeparam name="TStride">The buffer stride type.</typeparam>
        /// <param name="extent">The extent of the buffer.</param>
        /// <param name="stride">The buffer stride to use.</param>
        /// <returns>An allocated n-D buffer on the this accelerator.</returns>
        public ArrayView<T> AllocateRaw<T, TExtent, TStrideIndex, TStride>(
            TExtent extent,
            TStride stride)
            where T : unmanaged
            where TExtent : struct, ILongIndex<TExtent, TStrideIndex>
            where TStrideIndex : struct, IIntIndex<TStrideIndex, TExtent>
            where TStride : struct, IStride<TStrideIndex, TExtent>
        {
            EnsureBlittable<T>();

            // Determine the appropriate length to allocate
            int elementSize = ArrayView<T>.ElementSize;
            long length = stride.ComputeBufferLength(extent);

            // Allocate an unsafe buffer
            var buffer = AllocateRaw(length, elementSize);
            return new ArrayView<T>(buffer, 0L, length);
        }

        /// <summary>
        /// Ensures that the specified type <typeparamref name="T"/> is blittable.
        /// </summary>
        /// <typeparam name="T">The type to test.</typeparam>
        private void EnsureBlittable<T>()
            where T : unmanaged
        {
            // Check for blittable types
            var typeContext = Context.TypeContext;
            var elementType = typeof(T);
            var typeInfo = typeContext.GetTypeInfo(elementType);
            if (typeInfo.IsBlittable)
                return;

            throw new NotSupportedException(
                string.Format(
                    RuntimeErrorMessages.NotSupportedNonBlittableType,
                    elementType.GetStringRepresentation()));
        }

        #endregion

        #region Occupancy

        /// <summary>
        /// Estimates the occupancy of the given kernel with the given group size of a
        /// single multiprocessor.
        /// </summary>
        /// <typeparam name="TIndex">The index type of the group dimension.</typeparam>
        /// <param name="kernel">The kernel used for the estimation.</param>
        /// <param name="groupDim">The group dimension.</param>
        /// <returns>
        /// The estimated occupancy in percent [0.0, 1.0] of a single multiprocessor.
        /// </returns>
        public float EstimateOccupancyPerMultiprocessor<TIndex>(
            Kernel kernel,
            TIndex groupDim)
            where TIndex : struct, IIndex =>
            EstimateOccupancyPerMultiprocessor(kernel, groupDim.GetIntSize());

        /// <summary>
        /// Estimates the occupancy of the given kernel with the given group size of a
        /// single multiprocessor.
        /// </summary>
        /// <param name="kernel">The kernel used for the estimation.</param>
        /// <param name="groupSize">The number of threads per group.</param>
        /// <returns>
        /// The estimated occupancy in percent [0.0, 1.0] of a single multiprocessor.
        /// </returns>
        public float EstimateOccupancyPerMultiprocessor(Kernel kernel, int groupSize) =>
            EstimateOccupancyPerMultiprocessor(kernel, groupSize, 0);

        /// <summary>
        /// Estimates the occupancy of the given kernel with the given group size of a
        /// single multiprocessor.
        /// </summary>
        /// <param name="kernel">The kernel used for the estimation.</param>
        /// <param name="groupSize">The number of threads per group.</param>
        /// <param name="dynamicSharedMemorySizeInBytes">
        /// The required dynamic shared-memory size in bytes.
        /// </param>
        /// <returns>
        /// The estimated occupancy in percent [0.0, 1.0] of a single multiprocessor.
        /// </returns>
        public float EstimateOccupancyPerMultiprocessor(
            Kernel kernel,
            int groupSize,
            int dynamicSharedMemorySizeInBytes)
        {
            var maxActiveGroups = EstimateMaxActiveGroupsPerMultiprocessor(
                kernel,
                groupSize,
                dynamicSharedMemorySizeInBytes);
            return (maxActiveGroups * groupSize) / (float)MaxNumThreadsPerGroup;
        }

        /// <summary>
        /// Estimates the maximum number of active groups per multiprocessor for the
        /// given kernel.
        /// </summary>
        /// <typeparam name="TIndex">The index type of the group dimension.</typeparam>
        /// <param name="kernel">The kernel used for the computation of the maximum
        /// number of active groups.</param>
        /// <param name="groupDim">The group dimension.</param>
        /// <returns>
        /// The maximum number of active groups per multiprocessor for the given kernel.
        /// </returns>
        public int EstimateMaxActiveGroupsPerMultiprocessor<TIndex>(
            Kernel kernel,
            TIndex groupDim)
            where TIndex : struct, IIndex =>
            EstimateMaxActiveGroupsPerMultiprocessor(kernel, groupDim.GetIntSize());

        /// <summary>
        /// Estimates the maximum number of active groups per multiprocessor for the
        /// given kernel.
        /// </summary>
        /// <param name="kernel">The kernel used for the computation of the maximum
        /// number of active groups.</param>
        /// <param name="groupSize">The number of threads per group.</param>
        /// <returns>
        /// The maximum number of active groups per multiprocessor for the given kernel.
        /// </returns>
        public int EstimateMaxActiveGroupsPerMultiprocessor(
            Kernel kernel,
            int groupSize) =>
            EstimateMaxActiveGroupsPerMultiprocessor(kernel, groupSize, 0);

        /// <summary>
        /// Estimates the maximum number of active groups per multiprocessor for the
        /// given kernel.
        /// </summary>
        /// <param name="kernel">The kernel used for the computation of the maximum
        /// number of active groups.</param>
        /// <param name="groupSize">The number of threads per group.</param>
        /// <param name="dynamicSharedMemorySizeInBytes">
        /// The required dynamic shared-memory size in bytes.
        /// </param>
        /// <returns>
        /// The maximum number of active groups per multiprocessor for the given kernel.
        /// </returns>
        public int EstimateMaxActiveGroupsPerMultiprocessor(
            Kernel kernel,
            int groupSize,
            int dynamicSharedMemorySizeInBytes)
        {
            if (kernel == null)
                throw new ArgumentNullException(nameof(kernel));
            if (groupSize < 1)
                throw new ArgumentNullException(nameof(groupSize));
            if (dynamicSharedMemorySizeInBytes < 0)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(dynamicSharedMemorySizeInBytes));
            }
            Bind();
            return EstimateMaxActiveGroupsPerMultiprocessorInternal(
                kernel,
                groupSize,
                dynamicSharedMemorySizeInBytes);
        }

        /// <summary>
        /// Estimates the maximum number of active groups per multiprocessor for the
        /// given kernel.
        /// </summary>
        /// <param name="kernel">The kernel used for the computation of the maximum
        /// number of active groups.</param>
        /// <param name="groupSize">The number of threads per group.</param>
        /// <param name="dynamicSharedMemorySizeInBytes">
        /// The required dynamic shared-memory size in bytes.
        /// </param>
        /// <remarks>
        /// Note that the arguments do not have to be verified since they are already
        /// verified.
        /// </remarks>
        /// <returns>
        /// The maximum number of active groups per multiprocessor for the given kernel.
        /// </returns>
        protected abstract int EstimateMaxActiveGroupsPerMultiprocessorInternal(
            Kernel kernel,
            int groupSize,
            int dynamicSharedMemorySizeInBytes);

        /// <summary>
        /// Estimates a group size to gain maximum occupancy on this device.
        /// </summary>
        /// <param name="kernel">The kernel used for the estimation.</param>
        /// <returns>
        /// An estimated group size to gain maximum occupancy on this device.
        /// </returns>
        public int EstimateGroupSize(Kernel kernel) =>
            EstimateGroupSize(kernel, 0, 0, out var _);

        /// <summary>
        /// Estimates a group size to gain maximum occupancy on this device.
        /// </summary>
        /// <param name="kernel">The kernel used for the estimation.</param>
        /// <param name="minGridSize">
        /// The minimum grid size to gain maximum occupancy on this device.
        /// </param>
        /// <returns>
        /// An estimated group size to gain maximum occupancy on this device.
        /// </returns>
        public int EstimateGroupSize(Kernel kernel, out int minGridSize) =>
            EstimateGroupSize(kernel, 0, 0, out minGridSize);

        /// <summary>
        /// Estimates a group size to gain maximum occupancy on this device.
        /// </summary>
        /// <param name="kernel">The kernel used for the estimation.</param>
        /// <param name="dynamicSharedMemorySizeInBytes">
        /// The required dynamic shared-memory size in bytes.
        /// </param>
        /// <param name="minGridSize">
        /// The minimum grid size to gain maximum occupancy on this device.
        /// </param>
        /// <returns>
        /// An estimated group size to gain maximum occupancy on this device.
        /// </returns>
        public int EstimateGroupSize(
            Kernel kernel,
            int dynamicSharedMemorySizeInBytes,
            out int minGridSize) =>
            EstimateGroupSize(
                kernel,
                dynamicSharedMemorySizeInBytes,
                0,
                out minGridSize);

        /// <summary>
        /// Estimates a group size to gain maximum occupancy on this device.
        /// </summary>
        /// <param name="kernel">The kernel used for the estimation.</param>
        /// <param name="computeSharedMemorySize">
        /// A callback to compute the required amount of shared memory in bytes for a
        /// given group size.
        /// </param>
        /// <param name="minGridSize">
        /// The minimum grid size to gain maximum occupancy on this device.
        /// </param>
        /// <returns>
        /// An estimated group size to gain maximum occupancy on this device.
        /// </returns>
        public int EstimateGroupSize(
            Kernel kernel,
            Func<int, int> computeSharedMemorySize,
            out int minGridSize) =>
            EstimateGroupSize(
                kernel,
                computeSharedMemorySize,
                0,
                out minGridSize);

        /// <summary>
        /// Estimates a group size to gain maximum occupancy on this device.
        /// </summary>
        /// <param name="kernel">The kernel used for the estimation.</param>
        /// <param name="computeSharedMemorySize">
        /// A callback to compute the required amount of shared memory in bytes for a
        /// given group size.
        /// </param>
        /// <param name="maxGroupSize">
        /// The maximum group-size limit on a single multiprocessor.
        /// </param>
        /// <param name="minGridSize">
        /// The minimum grid size to gain maximum occupancy on this device.
        /// </param>
        /// <returns>
        /// An estimated group size to gain maximum occupancy on this device.
        /// </returns>
        public int EstimateGroupSize(
            Kernel kernel,
            Func<int, int> computeSharedMemorySize,
            int maxGroupSize,
            out int minGridSize)
        {
            if (kernel == null)
                throw new ArgumentNullException(nameof(kernel));
            if (computeSharedMemorySize == null)
                throw new ArgumentNullException(nameof(computeSharedMemorySize));
            if (maxGroupSize < 0)
                throw new ArgumentOutOfRangeException(nameof(maxGroupSize));
            Bind();
            return EstimateGroupSizeInternal(
                kernel,
                computeSharedMemorySize,
                maxGroupSize,
                out minGridSize);
        }

        /// <summary>
        /// Estimates a group size to gain maximum occupancy on this device.
        /// </summary>
        /// <param name="kernel">The kernel used for the estimation.</param>
        /// <param name="computeSharedMemorySize">
        /// A callback to compute the required amount of shared memory in bytes for a
        /// given group size.
        /// </param>
        /// <param name="maxGroupSize">
        /// The maximum group-size limit on a single multiprocessor.
        /// </param>
        /// <param name="minGridSize">
        /// The minimum grid size to gain maximum occupancy on this device.
        /// </param>
        /// <remarks>
        /// Note that the arguments do not have to be verified since they are already
        /// verified.
        /// </remarks>
        /// <returns>
        /// An estimated group size to gain maximum occupancy on this device.
        /// </returns>
        protected abstract int EstimateGroupSizeInternal(
            Kernel kernel,
            Func<int, int> computeSharedMemorySize,
            int maxGroupSize,
            out int minGridSize);

        /// <summary>
        /// Estimates a group size to gain maximum occupancy on this device.
        /// </summary>
        /// <param name="kernel">The kernel used for the estimation.</param>
        /// <param name="dynamicSharedMemorySizeInBytes">
        /// The required dynamic shared-memory size in bytes.
        /// </param>
        /// <param name="maxGroupSize">
        /// The maximum group-size limit on a single multiprocessor.
        /// </param>
        /// <param name="minGridSize">
        /// The minimum grid size to gain maximum occupancy on this device.
        /// </param>
        /// <returns>
        /// An estimated group size to gain maximum occupancy on this device.
        /// </returns>
        public int EstimateGroupSize(
            Kernel kernel,
            int dynamicSharedMemorySizeInBytes,
            int maxGroupSize,
            out int minGridSize)
        {
            if (kernel == null)
                throw new ArgumentNullException(nameof(kernel));
            if (maxGroupSize < 0)
                throw new ArgumentOutOfRangeException(nameof(maxGroupSize));
            if (dynamicSharedMemorySizeInBytes < 0)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(dynamicSharedMemorySizeInBytes));
            }
            Bind();
            return EstimateGroupSizeInternal(
                kernel,
                dynamicSharedMemorySizeInBytes,
                maxGroupSize,
                out minGridSize);
        }

        /// <summary>
        /// Estimates a group size to gain maximum occupancy on this device.
        /// </summary>
        /// <param name="kernel">The kernel used for the estimation.</param>
        /// <param name="dynamicSharedMemorySizeInBytes">
        /// The required dynamic shared-memory size in bytes.
        /// </param>
        /// <param name="maxGroupSize">
        /// The maximum group-size limit on a single multiprocessor.
        /// </param>
        /// <param name="minGridSize">
        /// The minimum grid size to gain maximum occupancy on this device.
        /// </param>
        /// <remarks>
        /// Note that the arguments do not have to be verified since they are already
        /// verified.
        /// </remarks>
        /// <returns>
        /// An estimated group size to gain maximum occupancy on this device.
        /// </returns>
        protected abstract int EstimateGroupSizeInternal(
            Kernel kernel,
            int dynamicSharedMemorySizeInBytes,
            int maxGroupSize,
            out int minGridSize);

        #endregion

        #region IDisposable

        /// <summary cref="DisposeBase.Dispose(bool)"/>
        protected sealed override void Dispose(bool disposing)
        {
            Debug.Assert(NativePtr != IntPtr.Zero, "Invalid native pointer");

            // Dispose all accelerator extensions
            base.Dispose(disposing);

            // The main disposal functionality is locked to avoid parallel dispose
            // operations that can happen due to a parallel invocation of the .Net GC
            lock (syncRoot)
            {
                // Invoke the disposal event
                Disposed?.Invoke(this, EventArgs.Empty);

                // Bind the current instance
                Bind();

                // Dispose all child objects
                DisposeChildObjects_SyncRoot(disposing);

                // Dispose the accelerator instance
                DisposeAccelerator_SyncRoot(disposing);

                // Wait for the GC thread to terminate
                DisposeGC_SyncRoot();

                // Unbind the current accelerator and reset it to no accelerator
                OnUnbind();
                currentAccelerator = null;

                // Clear the native pointer
                NativePtr = IntPtr.Zero;

                // Commit all changes
                Thread.MemoryBarrier();
            }
        }

        /// <summary>
        /// Disposes this accelerator instance (synchronized with the current main
        /// synchronization object of this accelerator).
        /// </summary>
        /// <param name="disposing">
        /// True, if the method is not called by the finalizer.
        /// </param>
        protected abstract void DisposeAccelerator_SyncRoot(bool disposing);

        #endregion

        #region Object

        /// <summary>
        /// Returns the string representation of this accelerator.
        /// </summary>
        /// <returns>The string representation of this accelerator.</returns>
        public override string ToString() =>
            $"{Name} [WarpSize: {WarpSize}, " +
            $"MaxNumThreadsPerGroup: {MaxNumThreadsPerGroup}, " +
            $"MemorySize: {MemorySize}]";

        #endregion
    }
}
