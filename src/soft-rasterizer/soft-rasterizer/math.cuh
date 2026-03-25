#pragma once
#include <cmath>

// Minimal linear algebra for a software rasterizer.
// All types are usable on both host and device.

namespace Math
{
    struct Vec3
    {
        float X, Y, Z;

        __host__ __device__ Vec3() : X(0), Y(0), Z(0) {}
        __host__ __device__ Vec3(float x, float y, float z) : X(x), Y(y), Z(z) {}

        __host__ __device__ auto operator+(Vec3 b) const -> Vec3 { return { X + b.X, Y + b.Y, Z + b.Z }; }
        __host__ __device__ auto operator-(Vec3 b) const -> Vec3 { return { X - b.X, Y - b.Y, Z - b.Z }; }
        __host__ __device__ auto operator*(float s) const -> Vec3 { return { X * s, Y * s, Z * s }; }
    };

    __host__ __device__ inline auto Dot(Vec3 a, Vec3 b) -> float
    {
        return a.X * b.X + a.Y * b.Y + a.Z * b.Z;
    }

    __host__ __device__ inline auto Cross(Vec3 a, Vec3 b) -> Vec3
    {
        return { a.Y * b.Z - a.Z * b.Y,
                 a.Z * b.X - a.X * b.Z,
                 a.X * b.Y - a.Y * b.X };
    }

    __host__ __device__ inline auto Normalize(Vec3 v) -> Vec3
    {
        float len = sqrtf(Dot(v, v));
        return len > 0.0f ? v * (1.0f / len) : Vec3{};
    }

    struct Vec4
    {
        float X, Y, Z, W;

        __host__ __device__ Vec4() : X(0), Y(0), Z(0), W(0) {}
        __host__ __device__ Vec4(float x, float y, float z, float w) : X(x), Y(y), Z(z), W(w) {}
        __host__ __device__ Vec4(Vec3 v, float w) : X(v.X), Y(v.Y), Z(v.Z), W(w) {}

        __host__ __device__ auto ToVec3() const -> Vec3 { return { X, Y, Z }; }

        // Perspective divide: clip space → NDC
        __host__ __device__ auto PerspectiveDivide() const -> Vec3
        {
            float invW = 1.0f / W;
            return { X * invW, Y * invW, Z * invW };
        }
    };

    // 4x4 column-major matrix (M[col][row])
    struct Mat4
    {
        float M[4][4];

        __host__ __device__ Mat4()
        {
            for (int c = 0; c < 4; c++)
                for (int r = 0; r < 4; r++)
                    M[c][r] = 0.0f;
        }

        __host__ __device__ static auto Identity() -> Mat4
        {
            Mat4 m;
            m.M[0][0] = m.M[1][1] = m.M[2][2] = m.M[3][3] = 1.0f;
            return m;
        }

        // Matrix * Vec4 (column vector)
        __host__ __device__ auto operator*(Vec4 v) const -> Vec4
        {
            return {
                M[0][0] * v.X + M[1][0] * v.Y + M[2][0] * v.Z + M[3][0] * v.W,
                M[0][1] * v.X + M[1][1] * v.Y + M[2][1] * v.Z + M[3][1] * v.W,
                M[0][2] * v.X + M[1][2] * v.Y + M[2][2] * v.Z + M[3][2] * v.W,
                M[0][3] * v.X + M[1][3] * v.Y + M[2][3] * v.Z + M[3][3] * v.W
            };
        }

        // Matrix * Matrix
        __host__ __device__ auto operator*(Mat4 b) const -> Mat4
        {
            Mat4 result;
            for (int c = 0; c < 4; c++)
                for (int r = 0; r < 4; r++)
                    result.M[c][r] = M[0][r] * b.M[c][0]
                                   + M[1][r] * b.M[c][1]
                                   + M[2][r] * b.M[c][2]
                                   + M[3][r] * b.M[c][3];
            return result;
        }
    };

    // Rotation around the Y axis by `angle` radians
    inline auto RotationY(float angle) -> Mat4
    {
        float c = cosf(angle);
        float s = sinf(angle);
        Mat4 m = Mat4::Identity();
        m.M[0][0] =  c;
        m.M[2][0] = -s;
        m.M[0][2] =  s;
        m.M[2][2] =  c;
        return m;
    }

    // Simple look-at view matrix (right-handed)
    inline auto LookAt(Vec3 eye, Vec3 target, Vec3 up) -> Mat4
    {
        Vec3 f = Normalize(target - eye);  // forward
        Vec3 r = Normalize(Cross(f, up));  // right
        Vec3 u = Cross(r, f);              // true up

        Mat4 m = Mat4::Identity();
        m.M[0][0] =  r.X;  m.M[1][0] =  r.Y;  m.M[2][0] =  r.Z;
        m.M[0][1] =  u.X;  m.M[1][1] =  u.Y;  m.M[2][1] =  u.Z;
        m.M[0][2] = -f.X;  m.M[1][2] = -f.Y;  m.M[2][2] = -f.Z;
        m.M[3][0] = -Dot(r, eye);
        m.M[3][1] = -Dot(u, eye);
        m.M[3][2] =  Dot(f, eye);
        return m;
    }

    // Perspective projection matrix (right-handed, depth mapped to [0, 1])
    inline auto Perspective(float fovYRadians, float aspect, float nearPlane, float farPlane) -> Mat4
    {
        float tanHalfFov = tanf(fovYRadians * 0.5f);
        Mat4 m;
        m.M[0][0] = 1.0f / (aspect * tanHalfFov);
        m.M[1][1] = 1.0f / tanHalfFov;
        m.M[2][2] = -(farPlane) / (farPlane - nearPlane);
        m.M[2][3] = -1.0f;
        m.M[3][2] = -(farPlane * nearPlane) / (farPlane - nearPlane);
        return m;
    }
}
