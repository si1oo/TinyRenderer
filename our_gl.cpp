#include <cmath>
#include <limits>
#include <cstdlib>
#include "our_gl.h"

Matrix ModelView;
Matrix Viewport;
Matrix Projection;

IShader::~IShader() {};

void viewport(int x, int y, int w, int h) {
    Viewport = Matrix::identity();
    Viewport[0][3] = x + w / 2.f;
    Viewport[1][3] = y + h / 2.f;
    Viewport[2][3] = depth / 2.f;
    Viewport[0][0] = w / 2.f;
    Viewport[1][1] = h / 2.f;
    Viewport[2][2] = depth / 2.f;
}

void projection(float coeff) { // coeff = -1/c
    Projection = Matrix::identity();
    Projection[3][2] = coeff;
}

void lookat(Vec3f eye, Vec3f center, Vec3f up) {
    Vec3f z = (eye - center).normalize();
    Vec3f x = cross(up, z).normalize();
    Vec3f y = cross(z, x).normalize();
    ModelView = Matrix::identity();
    for (int i = 0; i < 3; i++) {
        ModelView[0][i] = x[i];
        ModelView[1][i] = y[i];
        ModelView[2][i] = z[i];
        ModelView[i][3] = -center[i];
    }
}

Vec3f barycentric(Vec3f* pts, Vec3f P) {
    Vec3f u = cross(Vec3f(pts[2].x - pts[0].x, pts[1].x - pts[0].x, pts[0].x - P.x),
        Vec3f(pts[2].y - pts[0].y, pts[1].y - pts[0].y, pts[0].y - P.y));
    if (std::abs(u.z) < 1) return Vec3f(-1, 1, 1);
    return Vec3f(1.0f - (u.x + u.y) / u.z, u.y / u.z, u.x / u.z); 
}


void triangle(Vec4f* pts, IShader& shader, TGAImage& image, TGAImage& zbuffer) {
   //获取包围盒
    Vec2f bboxmin(std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
    Vec2f bboxmax(-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max());
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 2; j++) {
            bboxmin[j] = std::min(bboxmin[j], pts[i][j] / pts[i][3]);
            bboxmax[j] = std::max(bboxmax[j], pts[i][j] / pts[i][3]);
        }
    }
    Vec2i P;
    Vec3f* pts_points = new Vec3f[3]
    {
    Vec3f(pts[0][0] / pts[0][3],pts[0][1] / pts[0][3],pts[0][2] / pts[0][3]) ,
    Vec3f(pts[1][0] / pts[1][3],pts[1][1] / pts[1][3],pts[1][2] / pts[1][3]) ,
    Vec3f(pts[2][0] / pts[2][3],pts[2][1] / pts[2][3],pts[2][2] / pts[2][3]) ,
    };
    TGAColor color;

    for (P.x = bboxmin.x; P.x <= bboxmax.x; P.x++) {
        for (P.y = bboxmin.y; P.y <= bboxmax.y; P.y++) {
            Vec3f bary = barycentric(pts_points, Vec3f(P.x, P.y, 0)); //获取每个像素的重心坐标
            float z = pts[0][2] * bary.x + pts[1][2] * bary.y + pts[2][2] * bary.z; 
            float w = pts[0][3] * bary.x + pts[1][3] * bary.y + pts[2][3] * bary.z;

            int frag_depth = std::max(0, std::min(255, int(z / w + 0.5f)));//对深度进行四舍五入，实现精度修正

            //zbuffer的get会返回TGAColor，内部有一个数组，能够表示深度
            if (bary.x < 0 || bary.y < 0 || bary.z<0 || zbuffer.get(P.x, P.y)[0]>frag_depth) continue;
            bool discard = shader.fragment(bary, color);
            if (!discard) {
                //生成原图的同时维护一张深度图
                zbuffer.set(P.x, P.y, TGAColor(frag_depth));
                image.set(P.x, P.y, color);
            }
        }
    }
    delete[] pts_points;
}

