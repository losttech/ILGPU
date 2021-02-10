// ---------------------------------------------------------------------------------------
//                                        ILGPU
//                        Copyright (c) 2016-2020 Marcel Koester
//                                    www.ilgpu.net
//
// File: CPURuntimeThreadContext.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details
// ---------------------------------------------------------------------------------------

using System;
using System.Reflection;
using System.Runtime.CompilerServices;

namespace ILGPU.Runtime.CPU
{
    /// <summary>
    /// Represents a runtime context for single threads.
    /// </summary>
    static class CPURuntimeThreadContext
    {
        #region Fields

        /// <summary>
        /// Represents the <see cref="SetupIndices(Index3D, Index3D)"/> method.
        /// </summary>
        internal static MethodInfo SetupIndicesMethod =
            typeof(CPURuntimeThreadContext).GetMethod(
                nameof(SetupIndices),
                BindingFlags.NonPublic | BindingFlags.Static);

        #endregion

        #region Thread Static

        /// <summary>
        /// The grid index within the scheduled thread grid
        /// of the debug CPU accelerator.
        /// </summary>
        [ThreadStatic]
        private static Index3D gridIndexValue;

        /// <summary>
        /// The group index within the scheduled thread grid
        /// of the debug CPU accelerator.
        /// </summary>
        [ThreadStatic]
        private static Index3D groupIndexValue;

        /// <summary>
        /// The grid dimension within the scheduled thread grid
        /// of the debug CPU accelerator.
        /// </summary>
        [ThreadStatic]
        private static Index3D gridDimensionValue;

        /// <summary>
        /// The group dimension within the scheduled thread grid
        /// of the debug CPU accelerator.
        /// </summary>
        [ThreadStatic]
        private static Index3D groupDimensionValue;

        #endregion

        #region Properties

        /// <summary>
        /// Returns the grid index within the scheduled thread grid.
        /// </summary>
        public static Index3D GridIndex => gridIndexValue;

        /// <summary>
        /// Returns the group index within the scheduled thread grid.
        /// </summary>
        public static Index3D GroupIndex => groupIndexValue;

        /// <summary>
        /// Returns the group dimension of the scheduled thread grid.
        /// </summary>
        public static Index3D GridDimension => gridDimensionValue;

        /// <summary>
        /// Returns the group dimension of the scheduled thread grid.
        /// </summary>
        public static Index3D GroupDimension => groupDimensionValue;

        /// <summary>
        /// Returns the current total group size in number of threads.
        /// </summary>
        public static int GroupSize => GroupDimension.Size;

        #endregion

        #region Methods

        /// <summary>
        /// Setups the current grid and group indices.
        /// </summary>
        /// <param name="gridIndex">The grid index.</param>
        /// <param name="groupIndex">The group index.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static void SetupIndices(Index3D gridIndex, Index3D groupIndex)
        {
            gridIndexValue = gridIndex;
            groupIndexValue = groupIndex;
        }

        /// <summary>
        /// Setups the scheduled grid and group dimensions and resets
        /// the current grid and group indices.
        /// </summary>
        /// <param name="gridDimension">The grid dimension.</param>
        /// <param name="groupDimension">The group dimension.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static void SetupDimensions(
            Index3D gridDimension,
            Index3D groupDimension)
        {
            SetupIndices(default, default);
            gridDimensionValue = gridDimension;
            groupDimensionValue = groupDimension;
        }

        #endregion
    }
}
