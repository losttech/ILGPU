#nullable enable
using ILGPU.IR;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;

namespace ILGPU.Frontend
{
    public class SpecializedILMethod: IEquatable<SpecializedILMethod>
    {
        public MethodBase Method { get; private set; }
        public Value?[] ParameterValues { get; private set; }

        public bool Equals(SpecializedILMethod other)
        {
            if (other is null) return false;

            return Method.Equals(other.Method)
                && ParameterValues.Zip(other.ParameterValues, (t, o) => t is null && o is null || t.Equals(o))
                    .All(b => b);
        }
    }
}
