#include<vector>
#include <iostream>
#include "tgaimage.h"
#include "model.h"
#include "geometry.h"
#include "our_gl.h"

Model* model = NULL;
const int width = 800;
const int height = 800;

Vec3f light_dir(5, 0, 0);
Vec3f eye(0, 0, 3);
Vec3f center(0, 0, 0);
Vec3f up(0, 1, 0);

TGAImage shadowmap(width, height, TGAImage::GRAYSCALE);
TGAImage shadowimage(width, height, TGAImage::GRAYSCALE);
TGAImage image(width, height, TGAImage::RGB);
TGAImage zbuffer(width, height, TGAImage::GRAYSCALE);


//Gouraud Shading
struct GouraudShader : public IShader {
    Vec3f varying_intensity; 

    virtual Vec4f vertex(int iface, int nthvert)
    {
        Vec4f gl_Vertex = embed<4>(model->vert(iface, nthvert));
        varying_intensity[nthvert] = std::max(0.f, model->normal(iface, nthvert) * light_dir);
        return  Viewport * Projection * ModelView * gl_Vertex;
    }
  
    virtual bool fragment(Vec3f bar, TGAColor& color) 
    {
        float intensity = varying_intensity * bar; 
        color = TGAColor(255, 255, 255) * intensity; 
        return false;                              
    }
};

//���������Gouraud Shading
struct TextureGouraudShader : public IShader {
    Vec3f varying_intensity; 
    mat<2, 3, float> varying_uv; 

    virtual Vec4f vertex(int iface, int nthvert) 
    {
        Vec4f gl_Vertex = embed<4>(model->vert(iface, nthvert));
        varying_intensity[nthvert] = std::max(0.f, model->normal(iface, nthvert) * light_dir);
        varying_uv.set_col(nthvert, model->uv(iface, nthvert));
        return Viewport * Projection * ModelView * gl_Vertex; 
    }

    virtual bool fragment(Vec3f bar, TGAColor& color) 
    {
        float intensity = varying_intensity * bar;   
        Vec2f uv = varying_uv * bar;         
        color = model->diffuse(uv) * intensity; 
        return false;                            
    }
};

//Phong Shading
struct Phong : public IShader {
    mat<2, 3, float> varying_uv;
    mat<3, 3, float> varying_nrm;
    mat<4, 4, float> uniform_M;   //  Projection*ModelView
    mat<4, 4, float> uniform_MIT; // (Projection*ModelView).invert_transpose()

    virtual Vec4f vertex(int iface, int nthvert) 
    {
        Vec4f gl_Vertex = embed<4>(model->vert(iface, nthvert));
        varying_uv.set_col(nthvert, model->uv(iface, nthvert));
        varying_nrm.set_col(nthvert, proj<3>(uniform_MIT * embed<4>(model->normal(iface, nthvert), 0.f))); //��þֲ�����ϵ�µĶ��㷨��
        return Viewport * Projection * ModelView * gl_Vertex;
    }

    virtual bool fragment(Vec3f bar, TGAColor& color) 
    {
        Vec3f bn = (varying_nrm * bar).normalize();
        Vec2f uv = varying_uv * bar;
        float diff = std::max(0.f, bn * light_dir);
        color = model->diffuse(uv) * diff;
        return false;
    }
};

//Blinn-Phong Reflection Model With Shadow
struct BlinnPhongShader : public IShader {
    mat<2, 3, float> varying_uv;  
    mat<4, 4, float> uniform_shadow;
    mat<3, 3, float> varying_tri; //��¼ÿ��������������꣬���ڼ�����Ӱ
    mat<4, 4, float> uniform_M;   // Projection*ModelView
    mat<4, 4, float> uniform_MIT; // (Projection*ModelView).invert_transpose()

    virtual Vec4f vertex(int iface, int nthvert) 
    {
        Vec4f gl_Vertex = embed<4>(model->vert(iface, nthvert));
        varying_uv.set_col(nthvert, model->uv(iface, nthvert));
        varying_tri.set_col(nthvert, model->vert(iface, nthvert));
        return Viewport * Projection * ModelView * gl_Vertex;
    }

    virtual bool fragment(Vec3f bar, TGAColor& color) 
    {
        //������Ӱ
        Vec4f sb_p = uniform_shadow * embed<4>(varying_tri * bar);
        sb_p = sb_p / sb_p[3];
        float shadow = .3 + .7 * (shadowmap.get(sb_p[0], sb_p[1])[0] < sb_p[2] + 43.34); //���� z-fighting
        //�������
        Vec2f uv = varying_uv * bar;
        //�������
        Vec3f n = proj<3>(uniform_MIT * embed<4>(model->normal(uv))).normalize();
        Vec3f l = proj<3>(uniform_M * embed<4>(light_dir)).normalize();
        Vec3f h = (n + l).normalize(); //�������
        float spec = pow(std::max(h * n, 0.0f), model->specular(uv));
        float diff = std::max(0.f, n * l);
        TGAColor c = model->diffuse(uv);
        color = c;
        for (int i = 0; i < 3; i++) color[i] = std::min<float>(25.0f + c[i] * shadow * (diff + .6 * spec), 255);
        return false;
    }
};

//ȫ������ϵ�µķ�����ͼ
struct GlobalTextureNormalShader : public IShader {
    mat<2, 3, float> varying_uv;
    mat<4, 4, float> uniform_M;   //  Projection*ModelView
    mat<4, 4, float> uniform_MIT; // (Projection*ModelView).invert_transpose()

    virtual Vec4f vertex(int iface, int nthvert) 
    {
        Vec4f gl_Vertex = embed<4>(model->vert(iface, nthvert));
        varying_uv.set_col(nthvert, model->uv(iface, nthvert));
        return Viewport * Projection * ModelView * gl_Vertex;
    }

    virtual bool fragment(Vec3f bar, TGAColor& color) 
    {
        Vec2f uv = varying_uv * bar;
        Vec3f n = proj<3>(uniform_MIT * embed<4>(model->normal(uv))).normalize();//�������ת������
        Vec3f l = proj<3>(uniform_M * embed<4>(light_dir)).normalize();//�������ת����Ĺ�Դ
        float intensity = std::max(0.f, n * l);
        color = model->diffuse(uv) * intensity;
        return false;
    }
};

//���߿ռ��µķ�����ͼ
struct TangentShader : public IShader {
    mat<2, 3, float> varying_uv; 
    mat<3, 3, float> varying_nrm; 
    mat<4, 3, float> varying_tri; //�任����������
    mat<3, 3, float> tri;         //�任�����ͨ����
    mat<4, 4, float> uniform_M;   //  Projection*ModelView
    mat<4, 4, float> uniform_MIT; // (Projection*ModelView).invert_transpose()

    virtual Vec4f vertex(int iface, int nthvert) 
    {
        Vec4f gl_Vertex = Projection * ModelView * embed<4>(model->vert(iface, nthvert));
        varying_uv.set_col(nthvert, model->uv(iface, nthvert));
        varying_nrm.set_col(nthvert, proj<3>(uniform_MIT * embed<4>(model->normal(iface, nthvert), 0.f)));//��ö��㷨��
        varying_tri.set_col(nthvert, gl_Vertex);
        tri.set_col(nthvert, proj<3>(gl_Vertex / gl_Vertex[3]));
        return Viewport * gl_Vertex;
    }
    virtual bool fragment(Vec3f bar, TGAColor& color) 
    {
        Vec3f bn = (varying_nrm * bar).normalize(); //��ֵ��ľֲ��ռ䷨��
        Vec2f uv = varying_uv * bar; //��ֵ���uv����
        //�������AB,AC
        Vec3f AB = tri.col(1) - tri.col(0);
        Vec3f AC = tri.col(2) - tri.col(0);
        //��æ�uv
        Vec3f uv1 = Vec3f(varying_uv[0][1] - varying_uv[0][0], varying_uv[1][1] - varying_uv[1][0], 0);
        Vec3f uv2 = Vec3f(varying_uv[0][2] - varying_uv[0][0], varying_uv[1][2] - varying_uv[1][0], 0);
        //���TBN
        Vec3f T = (AB * uv2[1] - AC * uv1[1]) / (uv1[0] * uv2[1] - uv2[0] * uv1[1]);
        Vec3f B = (AC * uv1[0] - AB * uv2[0]) / (uv1[0] * uv2[1] - uv2[0] * uv1[1]);
        //������
        Vec3f t = (T - bn * (T * bn)).normalize();
        Vec3f b = (B - bn * (B * bn) - t * (B * t)).normalize();
        //���tbn����
        mat<3, 3, float> TBN_M;
        TBN_M.set_col(0, t);
        TBN_M.set_col(1, b);
        TBN_M.set_col(2, bn);
        //���߿ռ�ת����ϵ
        Vec3f n = (TBN_M * model->normal(uv)).normalize();//��������еķ���
        float diff = std::max(0.f, n * light_dir);
        color = model->diffuse(uv) * diff;
        return false;
    }
};

//Shadow Shading
struct DepthShader : public IShader {
    mat<3, 3, float> varying_tri; //�ֲ��ռ��ж�������

    virtual Vec4f vertex(int iface, int nthvert) 
    {
        Vec4f gl_Vertex = embed<4>(model->vert(iface, nthvert));
        gl_Vertex = Viewport * Projection * ModelView * gl_Vertex;
        varying_tri.set_col(nthvert, proj<3>(gl_Vertex / gl_Vertex[3]));
        return gl_Vertex;
    }

    virtual bool fragment(Vec3f bar, TGAColor& color) 
    {
        Vec3f p = varying_tri * bar;
        color = TGAColor(255, 255, 255) * (p.z / depth);
        return false;
    }
};

void Triangle(IShader& shader, TGAImage& image, TGAImage& buffer,const char* ImageName) 
{
    for (int i = 0; i < model->nfaces(); i++) 
    {
        Vec4f screen_coords[3];
        for (int j = 0; j < 3; j++) 
            screen_coords[j] = shader.vertex(i, j);

        triangle(screen_coords, shader, image, buffer);
    }
    image.flip_vertically();
    image.write_tga_file(ImageName);
}

int main(int argc, char** argv) {
    //���ģ��
    model = new Model("E://tinyrender2//t2//african_head.obj");
    //��ʼ����ɫ��
    DepthShader depth_shader;
    BlinnPhongShader shader;
    //���ڼ�¼��Ӱת������
    mat<4, 4, float> Shadow_Transform;

    
    lookat(light_dir, center, up);//Modelview
    viewport(width / 8, height / 8, width * 3 / 4, height * 3 / 4);//Viewport
    projection(0.f);//Projection
    Triangle(depth_shader, shadowimage, shadowmap, "Shadow.tga");
  
    Shadow_Transform = Viewport * Projection * ModelView; 
     
    lookat(eye, center, up);
    projection(-1.f / (eye - center).norm());
    shader.uniform_M = Viewport * Projection * ModelView;
    shader.uniform_MIT = shader.uniform_M.invert_transpose();
    shader.uniform_shadow = Shadow_Transform;
    light_dir.normalize();
    Triangle(shader, image, zbuffer, "outimage4.tga");
     
    zbuffer.flip_vertically();
    zbuffer.write_tga_file("zbuffer.tga");

    delete model;
    return 0;
}