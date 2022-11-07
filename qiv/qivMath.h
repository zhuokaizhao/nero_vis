/*
  libqiv: Quick Inspection of Vector fields
  Copyright (C)  2022 University of Chicago. All rights reserved.
*/

#ifndef QIV_MATH_HAS_BEEN_INCLUDED
#define QIV_MATH_HAS_BEEN_INCLUDED

/*
  All these macros obey these conventions:
  "V2": 2-vector, which is just a length-2 array of reals
  "V3": 3-vector, which is just a length-3 array of reals
  "V4": 4-vector, which is just a length-4 array of reals
  "M2": 2x2 matrix, which is really just a length-4 array:

  M[0]  M[1]
  M[2]  M[3]

  "M3": 3x3 matrix, which is really just a length-9 array:

  M[0]  M[1]  M[2]
  M[3]  M[4]  M[5]
  M[6]  M[7]  M[8]

  "M4": 4x4 matrix, which is really just a length-16 array:

  M[ 0]  M[ 1]  M[ 2]  M[ 3]
  M[ 4]  M[ 5]  M[ 6]  M[ 7]
  M[ 8]  M[ 9]  M[10]  M[11]
  M[12]  M[13]  M[14]  M[15]

  As with all C macros, you should assume (until you verify otherwise by
  scrutinizing the macro definition) that it is NOT safe to use a variable for
  more than one macro argument: these don't work "in place". For example,
  using "MV2_MUL(u, m, u)" will give incorrect results because "u" is passed
  twice, once as input (third arg) and once as output (first arg). With
  separate variables u, v, for the two vectors, you would write:
  "MV2_MUL(v, m, u); V2_COPY(u, v)".

  These macros don't start with a QIV prefix because the intent is
  that individual source files will #include this, but, it will not
  be #include'd in the qiv.h, which defines the interface to the
  library (because users of the library don't need these macros)
*/
/* clang-format off */
// returns the max or min of two things
#define MAX(A, B) ((A) > (B) ? (A) : (B))
#define MIN(A, B) ((A) < (B) ? (A) : (B))

// clamps value V to interval [A,B]
#define CLAMP(A, V, B) ((V) < (A)               \
                        ? (A)                   \
                        : ((V) > (B)            \
                           ? (B)                \
                           : (V)))

// sets elements of 2-vector V
#define V2_SET(V, A, B)                         \
    ((V)[0] = (A), (V)[1] = (B))

// sets elements of 2-vector V to NaN
#define V2_SET_NAN(V)                           \
    ((V)[0] = qivNan(0),                      \
     (V)[1] = qivNan(0))

// test if given 2-vector V isfinite() for all entries
#define V2_ISFINITE(V)                          \
    (isfinite((V)[0]) && isfinite((V)[1]))

// copies elements of 2-vector S to V
#define V2_COPY(V, S)                           \
    ((V)[0] = (S)[0],                           \
     (V)[1] = (S)[1])

// 2-vector S is sum of 2-vectors A and B
#define V2_ADD(S, A, B) V2_SET(S,               \
                               (A)[0] + (B)[0], \
                               (A)[1] + (B)[1])

// sets 3-vector V to difference A - B of 3-vectors A, B
#define V2_SUB(V, A, B)                         \
    ((V)[0] = (A)[0] - (B)[0],                  \
     (V)[1] = (A)[1] - (B)[1])

// 2-vector U gets scaling, by A, of 2-vector V
#define V2_SCALE(U, A, V)                       \
    V2_SET(U, (V)[0]*(A), (V)[1]*(A))

// dot product of two 2-vectors U and V
#define V2_DOT(U, V)                            \
    ((U)[0]*(V)[0] + (U)[1]*(V)[1])

// length of a 2-vector V
#define V2_LEN(V) (sqrt(V2_DOT((V),(V))))

/* sets 2-vector L to lerp of 2-vectors A and B, as
   controlled by W in [0,1] */
#define V2_LERP(L, A, B, W)                     \
    ((L)[0] = (1-(W))*(A)[0] + (W)*(B)[0],      \
     (L)[1] = (1-(W))*(A)[1] + (W)*(B)[1])

/* sets 2-vector V to normalized 2-vector V = U/|U|,
   using T to store length */
#define V2_NORM(V, U, T)                        \
    ((T) = V2_LEN(U), V2_SCALE(V, (real)1/(T), U))

// 2-vector U gets 2x2 matrix M times 2-vector V
#define MV2_MUL(U, M, V)                        \
    ((U)[0] = (M)[0]*(V)[0] + (M)[1]*(V)[1],    \
     (U)[1] = (M)[2]*(V)[0] + (M)[3]*(V)[1])

// sets elements of a 2x2 matrix M
#define M2_SET(M, m11, m12, m21, m22)           \
    ((M)[0] = (m11), (M)[1] = (m12),            \
     (M)[2] = (m21), (M)[3] = (m22))

// sets given 2x2 matrix M to identity
#define M2_SET_IDENT(M)                         \
    ((M)[0] = 1, (M)[1] = 0,                    \
     (M)[2] = 0, (M)[3] = 1)                    \

// copies elements of 2x2 matrix S to D
#define M2_COPY(D, S)                           \
    ((D)[0] = (S)[0], (D)[1] = (S)[1],          \
     (D)[2] = (S)[2], (D)[3] = (S)[3])

// 2x2 matrix S is sum of 2x2 matrices A and B
#define M2_ADD(S, A, B)                                 \
    ((S)[0] = (A)[0]+(B)[0], (S)[1] = (A)[1]+(B)[1],    \
     (S)[2] = (A)[2]+(B)[2], (S)[3] = (A)[3]+(B)[3])

// determinant of 2x2 matrix M
#define _M2_DET(a,b,c,d) ((a)*(d) - (b)*(c))
#define M2_DET(M)                               \
    _M2_DET((M)[0], (M)[1], (M)[2], (M)[3])

// the Frobenius norm of 2x2 matrix M (the L2 norm of the 4-vector M)
#define M2_FROB(M)                           \
    sqrt((M)[ 0]*(M)[ 0] + (M)[ 1]*(M)[ 1] + \
         (M)[ 2]*(M)[ 2] + (M)[ 3]*(M)[ 3])

/* sets 2x2 matrix I to inverse-transpose of the
   upper 2x2 submatrix of 3x3
   matrix M, with the help of tmp variable T */
#define M23_INVERSE_TRANSPOSE(I, M, T)          \
    ((T) = (M)[0]*(M)[4] - (M)[1]*(M)[3],       \
     (I)[0] =  (M)[4]/T, (I)[2] = -(M)[1]/T,    \
     (I)[1] = -(M)[3]/T, (I)[3] =  (M)[0]/T)

// sets elements of a 3-vector V
#define V3_SET(V, A, B, C)                      \
    ((V)[0] = (A),                              \
     (V)[1] = (B),                              \
     (V)[2] = (C))

// sets elements of 3-vector V to NaN
#define V3_SET_NAN(V)                           \
    ((V)[0] = qivNan(0),                      \
     (V)[1] = qivNan(0),                      \
     (V)[2] = qivNan(0))

// test if given 3-vector V isfinite() for all entries
#define V3_ISFINITE(V)                          \
    (isfinite((V)[0]) &&                        \
     isfinite((V)[1]) &&                        \
     isfinite((V)[2]))

// copies elements of 3-vector S to V
#define V3_COPY(V, S)                           \
    ((V)[0] = (S)[0],                           \
     (V)[1] = (S)[1],                           \
     (V)[2] = (S)[2])

// 3-vector S gets sum of 3-vectors A and B
#define V3_ADD(S, A, B) V3_SET(S,               \
                               (A)[0] + (B)[0], \
                               (A)[1] + (B)[1], \
                               (A)[2] + (B)[2])

// sets 3-vector V to difference A - B of 3-vectors A, B
#define V3_SUB(V, A, B) V3_SET(V,               \
                               (A)[0] - (B)[0], \
                               (A)[1] - (B)[1], \
                               (A)[2] - (B)[2])

// 3-vector U gets scaling, by A, of 3-vector V
#define V3_SCALE(U, A, V)                               \
    V3_SET(U, (V)[0]*(A), (V)[1]*(A), (V)[2]*(A))

// 3-vector U gets sum of 3-vectors S*A and T*B, w/ scalars S, T
#define V3_SCALE_ADD(U, S, A, T, B) V3_SET(U,                           \
                                           (S)*(A)[0] + (T)*(B)[0],     \
                                           (S)*(A)[1] + (T)*(B)[1],     \
                                           (S)*(A)[2] + (T)*(B)[2])

// dot product of two 3-vectors U and V
#define V3_DOT(U, V)                            \
    ((U)[0]*(V)[0] +                            \
     (U)[1]*(V)[1] +                            \
     (U)[2]*(V)[2])

// length of a 3-vector V
#define V3_LEN(V) (sqrt(V3_DOT((V),(V))))

/* sets 3-vector V to normalized 3-vector V = U/|U|,
   using temporary variable T to store length */
#define V3_NORM(V, U, T)                                \
    ((T) = V3_LEN(U), V3_SCALE(V, (real)1/(T), U))

/* 3-vector C gets cross product AxB of A and B
   assuming coordinates are in a right-handed frame */
#define V3_CROSS(C, A, B)                       \
    ((C)[0] = (A)[1]*(B)[2] - (A)[2]*(B)[1],    \
     (C)[1] = (A)[2]*(B)[0] - (A)[0]*(B)[2],    \
     (C)[2] = (A)[0]*(B)[1] - (A)[1]*(B)[0])

/* sets 3-vector L to lerp of 3-vectors A and B, as
   controlled by W in [0,1] */
#define V3_LERP(L, A, B, W)                     \
    ((L)[0] = (1-(W))*(A)[0] + (W)*(B)[0],      \
     (L)[1] = (1-(W))*(A)[1] + (W)*(B)[1],      \
     (L)[2] = (1-(W))*(A)[2] + (W)*(B)[2])

// 3-vector U = 3x3 matrix M time 3-vector V
#define MV3_MUL(U, M, V)                                        \
    ((U)[0] = (M)[0]*(V)[0] + (M)[1]*(V)[1] + (M)[2]*(V)[2],    \
     (U)[1] = (M)[3]*(V)[0] + (M)[4]*(V)[1] + (M)[5]*(V)[2],    \
     (U)[2] = (M)[6]*(V)[0] + (M)[7]*(V)[1] + (M)[8]*(V)[2])

// sets elements of a 3x3 matrix M
#define M3_SET(M, m11, m12, m13, m21, m22, m23, m31, m32, m33)  \
    ((M)[0] = (m11), (M)[1] = (m12), (M)[2] = (m13),            \
     (M)[3] = (m21), (M)[4] = (m22), (M)[5] = (m23),            \
     (M)[6] = (m31), (M)[7] = (m32), (M)[8] = (m33))

// sets given 3x3 matrix M to identity
#define M3_SET_IDENT(M)                         \
    ((M)[0] = 1, (M)[1] = 0, (M)[2] = 0,        \
     (M)[3] = 0, (M)[4] = 1, (M)[5] = 0,        \
     (M)[6] = 0, (M)[7] = 0, (M)[8] = 1)

// sets given 3x3 matrix M to all NaNs
#define M3_SET_NAN(M)                                                   \
    ((M)[0] = qivNan(0), (M)[1] = qivNan(0), (M)[2] = qivNan(0),  \
     (M)[3] = qivNan(0), (M)[4] = qivNan(0), (M)[5] = qivNan(0),  \
     (M)[6] = qivNan(0), (M)[7] = qivNan(0), (M)[8] = qivNan(0))

// test if given 3x3 matrix M isfinite() for all entries
#define M3_ISFINITE(M)                                                  \
    (isfinite((M)[0]) && isfinite((M)[1]) && isfinite((M)[2]) &&        \
     isfinite((M)[3]) && isfinite((M)[4]) && isfinite((M)[5]) &&        \
     isfinite((M)[6]) && isfinite((M)[7]) && isfinite((M)[8]))

// copies elements of a 3x3 matrix
#define M3_COPY(M, S)                                   \
    ((M)[0] = (S)[0], (M)[1] = (S)[1], (M)[2] = (S)[2], \
     (M)[3] = (S)[3], (M)[4] = (S)[4], (M)[5] = (S)[5], \
     (M)[6] = (S)[6], (M)[7] = (S)[7], (M)[8] = (S)[8])

// determinant of 3x3 matrix
#define _M3_DET(a,b,c,d,e,f,g,h,i)            \
    (  (a)*(e)*(i)                            \
     + (d)*(h)*(c)                            \
     + (g)*(b)*(f)                            \
     - (g)*(e)*(c)                            \
     - (d)*(b)*(i)                            \
     - (a)*(h)*(f))
#define M3_DET(M) _M3_DET((M)[0],(M)[1],(M)[2],         \
                          (M)[3],(M)[4],(M)[5],         \
                          (M)[6],(M)[7],(M)[8])

// sets 3x3 I to inverse of 3x3 M, using tmp variable TMP
#define M3_INVERSE(I, M, TMP)                               \
    ((TMP) = M3_DET(M),                                     \
     (I)[0] =  _M2_DET((M)[4],(M)[5],(M)[7],(M)[8])/(TMP),  \
     (I)[1] = -_M2_DET((M)[1],(M)[2],(M)[7],(M)[8])/(TMP),  \
     (I)[2] =  _M2_DET((M)[1],(M)[2],(M)[4],(M)[5])/(TMP),  \
     (I)[3] = -_M2_DET((M)[3],(M)[5],(M)[6],(M)[8])/(TMP),  \
     (I)[4] =  _M2_DET((M)[0],(M)[2],(M)[6],(M)[8])/(TMP),  \
     (I)[5] = -_M2_DET((M)[0],(M)[2],(M)[3],(M)[5])/(TMP),  \
     (I)[6] =  _M2_DET((M)[3],(M)[4],(M)[6],(M)[7])/(TMP),  \
     (I)[7] = -_M2_DET((M)[0],(M)[1],(M)[6],(M)[7])/(TMP),  \
     (I)[8] =  _M2_DET((M)[0],(M)[1],(M)[3],(M)[4])/(TMP))

// sets 3x3 m2 to transpose of 3x3 m1, but DOES NOT work in-place!
#define M3_TRANSPOSE(m2, m1)                    \
    ((m2)[0] = (m1)[0],                         \
     (m2)[1] = (m1)[3],                         \
     (m2)[2] = (m1)[6],                         \
     (m2)[3] = (m1)[1],                         \
     (m2)[4] = (m1)[4],                         \
     (m2)[5] = (m1)[7],                         \
     (m2)[6] = (m1)[2],                         \
     (m2)[7] = (m1)[5],                         \
     (m2)[8] = (m1)[8])

// 3x3 matrix AB gets product of 3x3 matrices A and B (in that order)
#define M3_MUL(AB, A, B)                                          \
    ((AB)[0] = (A)[0]*(B)[0] + (A)[1]*(B)[3] + (A)[2]*(B)[6],     \
     (AB)[1] = (A)[0]*(B)[1] + (A)[1]*(B)[4] + (A)[2]*(B)[7],     \
     (AB)[2] = (A)[0]*(B)[2] + (A)[1]*(B)[5] + (A)[2]*(B)[8],     \
                                                                  \
     (AB)[3] = (A)[3]*(B)[0] + (A)[4]*(B)[3] + (A)[5]*(B)[6],     \
     (AB)[4] = (A)[3]*(B)[1] + (A)[4]*(B)[4] + (A)[5]*(B)[7],     \
     (AB)[5] = (A)[3]*(B)[2] + (A)[4]*(B)[5] + (A)[5]*(B)[8],     \
                                                                  \
     (AB)[6] = (A)[6]*(B)[0] + (A)[7]*(B)[3] + (A)[8]*(B)[6],     \
     (AB)[7] = (A)[6]*(B)[1] + (A)[7]*(B)[4] + (A)[8]*(B)[7],     \
     (AB)[8] = (A)[6]*(B)[2] + (A)[7]*(B)[5] + (A)[8]*(B)[8])

// the Frobenius norm of 3x3 matrix M (the L2 norm of the 9-vector M)
#define M3_FROB(M)                                             \
    sqrt((M)[ 0]*(M)[ 0] + (M)[ 1]*(M)[ 1] + (M)[ 2]*(M)[ 2] + \
         (M)[ 3]*(M)[ 3] + (M)[ 4]*(M)[ 4] + (M)[ 5]*(M)[ 5] + \
         (M)[ 6]*(M)[ 6] + (M)[ 7]*(M)[ 7] + (M)[ 8]*(M)[ 8])

// sets 3x3 matrix U to upper diagonal 3x3 sub-matrix of 4x4 matrix M
#define M34_UPPER(U, M)                         \
    M3_SET((U),                                 \
           (M)[ 0], (M)[ 1], (M)[ 2],           \
           (M)[ 4], (M)[ 5], (M)[ 6],           \
           (M)[ 8], (M)[ 9], (M)[10])

// sets elements of a 4-vector V
#define V4_SET(V, A, B, C, D)                                   \
    ((V)[0] = (A), (V)[1] = (B), (V)[2] = (C), (V)[3] = (D))

// sets elements of 4-vector V to NaN
#define V4_SET_NAN(V)                           \
    ((V)[0] = qivNan(0),                      \
     (V)[1] = qivNan(0),                      \
     (V)[2] = qivNan(0),                      \
     (V)[3] = qivNan(0))

// test if given 4-vector V isfinite() for all entries
#define V4_ISFINITE(V)                                                  \
    (isfinite((V)[0]) && isfinite((V)[1]) && isfinite((V)[2]) && isfinite((V)[3]))

// copies elements of 4-vector S to V
#define V4_COPY(V, S)                                                   \
    ((V)[0] = (S)[0], (V)[1] = (S)[1], (V)[2] = (S)[2], (V)[3] = (S)[3])

// 4-vector S gets sum of 3-vectors A and B
#define V4_ADD(S, A, B) V4_SET(S,               \
                               (A)[0] + (B)[0], \
                               (A)[1] + (B)[1], \
                               (A)[2] + (B)[2], \
                               (A)[3] + (B)[3])

// sets 3-vector V to difference A - B of 3-vectors A, B
#define V4_SUB(V, A, B) V4_SET(V,               \
                               (A)[0] - (B)[0], \
                               (A)[1] - (B)[1], \
                               (A)[2] - (B)[2], \
                               (A)[3] - (B)[3])

// 4-vector U is scaling, by A, of 4-vector V
#define V4_SCALE(U, A, V)                                       \
    V4_SET(U, (V)[0]*(A), (V)[1]*(A), (V)[2]*(A), (V)[3]*(A))

// dot product of two 4-vectors U and V
#define V4_DOT(U, V)                                                    \
    ((U)[0]*(V)[0] + (U)[1]*(V)[1] + (U)[2]*(V)[2] + (U)[3]*(V)[3])

// length of a 4-vector V
#define V4_LEN(V) (sqrt(V4_DOT((V),(V))))

/* sets 4-vector V to normalized 4-vector V = U/|U|;
   using T to store length */
#define V4_NORM(V, U, T)                                \
    ((T) = V4_LEN(U), V4_SCALE(V, (real)1/(T), U))

// 4-vector U = 4x4 matrix M times 4-vector V
#define MV4_MUL(U, M, V)                                                \
    ((U)[0] = (M)[ 0]*(V)[0] + (M)[ 1]*(V)[1] + (M)[ 2]*(V)[2] + (M)[ 3]*(V)[3], \
     (U)[1] = (M)[ 4]*(V)[0] + (M)[ 5]*(V)[1] + (M)[ 6]*(V)[2] + (M)[ 7]*(V)[3], \
     (U)[2] = (M)[ 8]*(V)[0] + (M)[ 9]*(V)[1] + (M)[10]*(V)[2] + (M)[11]*(V)[3], \
     (U)[3] = (M)[12]*(V)[0] + (M)[13]*(V)[1] + (M)[14]*(V)[2] + (M)[15]*(V)[3])

// sets 4x4 m2 to transpose of 4x4 m1, but DOES NOT work in-place!
#define M4_TRANSPOSE(m2, m1)                      \
    ((m2)[ 0] = (m1)[ 0],                         \
     (m2)[ 1] = (m1)[ 4],                         \
     (m2)[ 2] = (m1)[ 8],                         \
     (m2)[ 3] = (m1)[12],                         \
     (m2)[ 4] = (m1)[ 1],                         \
     (m2)[ 5] = (m1)[ 5],                         \
     (m2)[ 6] = (m1)[ 9],                         \
     (m2)[ 7] = (m1)[13],                         \
     (m2)[ 8] = (m1)[ 2],                         \
     (m2)[ 9] = (m1)[ 6],                         \
     (m2)[10] = (m1)[10],                         \
     (m2)[11] = (m1)[14],                         \
     (m2)[12] = (m1)[ 3],                         \
     (m2)[13] = (m1)[ 7],                         \
     (m2)[14] = (m1)[11],                         \
     (m2)[15] = (m1)[15])

// sets 4x4 AB to product of 3x3 matrices A and B (in that order)
#define M4_MUL(AB, A, B)                                                \
    ((AB)[ 0]=(A)[ 0]*(B)[ 0]+(A)[ 1]*(B)[ 4]+(A)[ 2]*(B)[ 8]+(A)[ 3]*(B)[12], \
     (AB)[ 1]=(A)[ 0]*(B)[ 1]+(A)[ 1]*(B)[ 5]+(A)[ 2]*(B)[ 9]+(A)[ 3]*(B)[13], \
     (AB)[ 2]=(A)[ 0]*(B)[ 2]+(A)[ 1]*(B)[ 6]+(A)[ 2]*(B)[10]+(A)[ 3]*(B)[14], \
     (AB)[ 3]=(A)[ 0]*(B)[ 3]+(A)[ 1]*(B)[ 7]+(A)[ 2]*(B)[11]+(A)[ 3]*(B)[15], \
                                                                        \
     (AB)[ 4]=(A)[ 4]*(B)[ 0]+(A)[ 5]*(B)[ 4]+(A)[ 6]*(B)[ 8]+(A)[ 7]*(B)[12], \
     (AB)[ 5]=(A)[ 4]*(B)[ 1]+(A)[ 5]*(B)[ 5]+(A)[ 6]*(B)[ 9]+(A)[ 7]*(B)[13], \
     (AB)[ 6]=(A)[ 4]*(B)[ 2]+(A)[ 5]*(B)[ 6]+(A)[ 6]*(B)[10]+(A)[ 7]*(B)[14], \
     (AB)[ 7]=(A)[ 4]*(B)[ 3]+(A)[ 5]*(B)[ 7]+(A)[ 6]*(B)[11]+(A)[ 7]*(B)[15], \
                                                                        \
     (AB)[ 8]=(A)[ 8]*(B)[ 0]+(A)[ 9]*(B)[ 4]+(A)[10]*(B)[ 8]+(A)[11]*(B)[12], \
     (AB)[ 9]=(A)[ 8]*(B)[ 1]+(A)[ 9]*(B)[ 5]+(A)[10]*(B)[ 9]+(A)[11]*(B)[13], \
     (AB)[10]=(A)[ 8]*(B)[ 2]+(A)[ 9]*(B)[ 6]+(A)[10]*(B)[10]+(A)[11]*(B)[14], \
     (AB)[11]=(A)[ 8]*(B)[ 3]+(A)[ 9]*(B)[ 7]+(A)[10]*(B)[11]+(A)[11]*(B)[15], \
                                                                        \
     (AB)[12]=(A)[12]*(B)[ 0]+(A)[13]*(B)[ 4]+(A)[14]*(B)[ 8]+(A)[15]*(B)[12], \
     (AB)[13]=(A)[12]*(B)[ 1]+(A)[13]*(B)[ 5]+(A)[14]*(B)[ 9]+(A)[15]*(B)[13], \
     (AB)[14]=(A)[12]*(B)[ 2]+(A)[13]*(B)[ 6]+(A)[14]*(B)[10]+(A)[15]*(B)[14], \
     (AB)[15]=(A)[12]*(B)[ 3]+(A)[13]*(B)[ 7]+(A)[14]*(B)[11]+(A)[15]*(B)[15])

// sets elements of a 4x4 matrix M
#define M4_SET(M,                                                       \
               m11, m12, m13, m14,                                      \
               m21, m22, m23, m24,                                      \
               m31, m32, m33, m34,                                      \
               m41, m42, m43, m44)                                      \
    ((M)[ 0] = (m11), (M)[ 1] = (m12), (M)[ 2] = (m13), (M)[ 3] = (m14), \
     (M)[ 4] = (m21), (M)[ 5] = (m22), (M)[ 6] = (m23), (M)[ 7] = (m24), \
     (M)[ 8] = (m31), (M)[ 9] = (m32), (M)[10] = (m33), (M)[11] = (m34), \
     (M)[12] = (m41), (M)[13] = (m42), (M)[14] = (m43), (M)[15] = (m44))

// sets given 4x4 matrix M to all NaNs
#define M4_SET_NAN(M)                                                   \
    ((M)[ 0] = qivNan(0), (M)[ 1] = qivNan(0), (M)[ 2] = qivNan(0), (M)[ 3] = qivNan(0), \
     (M)[ 4] = qivNan(0), (M)[ 5] = qivNan(0), (M)[ 6] = qivNan(0), (M)[ 7] = qivNan(0), \
     (M)[ 8] = qivNan(0), (M)[ 9] = qivNan(0), (M)[10] = qivNan(0), (M)[11] = qivNan(0), \
     (M)[12] = qivNan(0), (M)[13] = qivNan(0), (M)[14] = qivNan(0), (M)[15] = qivNan(0))

// test if given 4x4 matrix M isfinite() for all entries
#define M4_ISFINITE(M)                                                  \
    (isfinite((M)[ 0]) && isfinite((M)[ 1]) && isfinite((M)[ 2]) && isfinite((M)[ 3]) && \
     isfinite((M)[ 4]) && isfinite((M)[ 5]) && isfinite((M)[ 6]) && isfinite((M)[ 7]) && \
     isfinite((M)[ 8]) && isfinite((M)[ 9]) && isfinite((M)[10]) && isfinite((M)[11]) && \
     isfinite((M)[12]) && isfinite((M)[13]) && isfinite((M)[14]) && isfinite((M)[15]))

// copies 4x4 matrix S into D
#define M4_COPY(D, S)                           \
    M4_SET((D),                                 \
           (S)[ 0], (S)[ 1], (S)[ 2], (S)[ 3],  \
           (S)[ 4], (S)[ 5], (S)[ 6], (S)[ 7],  \
           (S)[ 8], (S)[ 9], (S)[10], (S)[11],  \
           (S)[12], (S)[13], (S)[14], (S)[15])

// the Frobenius norm of 4x4 matrix M (the L2 norm of the 16-vector M)
#define M4_FROB(M)                                                      \
    sqrt((M)[ 0]*(M)[ 0] + (M)[ 1]*(M)[ 1] + (M)[ 2]*(M)[ 2] + (M)[ 3]*(M)[ 3] + \
         (M)[ 4]*(M)[ 4] + (M)[ 5]*(M)[ 5] + (M)[ 6]*(M)[ 6] + (M)[ 7]*(M)[ 7] + \
         (M)[ 8]*(M)[ 8] + (M)[ 9]*(M)[ 9] + (M)[10]*(M)[10] + (M)[11]*(M)[11] + \
         (M)[12]*(M)[12] + (M)[13]*(M)[13] + (M)[14]*(M)[14] + (M)[15]*(M)[15])

// sets elements of a 6-vector V
#define V6_SET(V, A, B, C, D, E, F)             \
    ((V)[0] = (A), (V)[1] = (B), (V)[2] = (C),  \
     (V)[3] = (D), (V)[4] = (E), (V)[5] = (F))

// copies elements of 6-vector S to V
#define V6_COPY(V, S)                                   \
    ((V)[0] = (S)[0], (V)[1] = (S)[1], (V)[2] = (S)[2], \
     (V)[3] = (S)[3], (V)[4] = (S)[4], (V)[5] = (S)[5])
/* clang-format on */

#endif /* QIV_MATH_HAS_BEEN_INCLUDED */
