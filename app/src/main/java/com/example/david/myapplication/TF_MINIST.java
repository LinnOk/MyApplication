package com.example.david.myapplication;

/**
 * Created by david on 12/3/17.
 */

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.graphics.Matrix;
import android.os.Trace;
import android.util.Log;
import android.widget.ImageView;
import android.widget.TextView;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.ByteArrayOutputStream;
import java.lang.reflect.Array;


public class TF_MINIST {
    private static final String MODEL_FILE = "file:///android_asset/grf.pb"; //模型存放路径

    //数据的维度
    private static final int HEIGHT = 28;
    private static final int WIDTH = 28;
    private static final int MAXL = 10;

    //模型中输出变量的名称
    private static final String inputName = "Mul";
    //用于存储的模型输入数据
    private float[] inputs = new float[HEIGHT * WIDTH];

    //模型中输出变量的名称
    private static final String outputName = "final_result";
    //用于存储模型的输出数据,0-9
    private float[] outputs = new float[MAXL];



    TensorFlowInferenceInterface inferenceInterface;


    static {
        //加载库文件
        System.loadLibrary("tensorflow_inference");
    }

    TF_MINIST(AssetManager assetManager) {
        //接口定义
        inferenceInterface = new TensorFlowInferenceInterface(assetManager,MODEL_FILE);
    }
    /**
     * 将彩色图转换为灰度图
     * @param img 位图
     * @return  返回转换好的位图
     */
    public Bitmap convertGreyImg(Bitmap img) {
        int width = img.getWidth();         //获取位图的宽
        int height = img.getHeight();       //获取位图的高

        int []pixels = new int[width * height]; //通过位图的大小创建像素点数组

        img.getPixels(pixels, 0, width, 0, 0, width, height);
        int alpha = 0xFF << 24;
        for(int i = 0; i < height; i++)  {
            for(int j = 0; j < width; j++) {
                int grey = pixels[width * i + j];

                int red = ((grey  & 0x00FF0000 ) >> 16);
                int green = ((grey & 0x0000FF00) >> 8);
                int blue = (grey & 0x000000FF);

                grey = (int)((float) red * 0.3 + (float)green * 0.59 + (float)blue * 0.11);
                grey = alpha | (grey << 16) | (grey << 8) | grey;
                pixels[width * i + j] = grey;
            }
        }
        Bitmap result = Bitmap.createBitmap(width, height, Bitmap.Config.RGB_565);
        result.setPixels(pixels, 0, width, 0, 0, width, height);
        return result;
    }

    //将int数组转化为float数组
    public float[] ints2float(int[] src,int w){
        float res[]=new float[w];
        for(int i=0;i<w;++i) {
            res[i]=src[i];
        }
        return  res;
    }

    //返回数组中最大值的索引
    public int argmax(float output[]){
        int maxIndex=0;
        for(int i=1;i<MAXL;++i){
            maxIndex=output[i]>output[maxIndex]? i: maxIndex;
        }

        return maxIndex;


    }

    //将图像像素数据转为一维数组，isReverse判断是否需要反化图像
    public int[] getGrayPix_R(Bitmap bp,boolean isReverse){
        int[]pxs=new int[784];
        int acc=0;
        for(int m=0;m<28;++m){
            for(int n=0;n<28;++n){
                if(isReverse)
                    pxs[acc]=255-Color.red(bp.getPixel(n,m));
                else
                    pxs[acc]=Color.red(bp.getPixel(n,m));
                Log.d("12","gray_"+acc+":"+pxs[acc]+"_");
                ++acc;
            }
        }
        return pxs;

    }



    //灰化图像
    public Bitmap gray(Bitmap bitmap, int schema)
    {
        Bitmap bm = Bitmap.createBitmap(bitmap.getWidth(),bitmap.getHeight(), bitmap.getConfig());
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        for(int row=0; row<height; row++){
            for(int col=0; col<width; col++){
                int pixel = bitmap.getPixel(col, row);// ARGB
                int red = Color.red(pixel); // same as (pixel >> 16) &0xff
                int green = Color.green(pixel); // same as (pixel >> 8) &0xff
                int blue = Color.blue(pixel); // same as (pixel & 0xff)
                int alpha = Color.alpha(pixel); // same as (pixel >>> 24)
                int gray = 0;
                if(schema == 0)
                {
                    gray = (Math.max(blue, Math.max(red, green)) +
                            Math.min(blue, Math.min(red, green))) / 2;
                }
                else if(schema == 1)
                {
                    gray = (red + green + blue) / 3;
                }
                else if(schema == 2)
                {
                    gray = (int)(0.3 * red + 0.59 * green + 0.11 * blue);
                }
                Log.d("12","gray:"+gray);
                bm.setPixel(col, row, Color.argb(alpha, gray, gray, gray));
            }
        }
        return bm;
    }

    //获得预测结果
    public int  getAddResult(Bitmap bitmap) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        float scaleWidth = ((float)WIDTH) / width;
        float scaleHeight = ((float) HEIGHT) / height;
        Matrix matrix = new Matrix();

        //调整图像大小为28x28
        matrix.postScale(scaleWidth, scaleHeight);
        Bitmap newbm = Bitmap.createBitmap(bitmap, 0, 0, width, height, matrix, true);
        //灰化图片,注意这里虽然是灰化，但只是将R,G,B的值都变成一样的，所以本质上还是RGB的三通道图像
        newbm=gray(newbm,2);
        //这里的isReverse,true则获得反化的图像数据,否则不是,返回为一维数组
        int pxs[]=getGrayPix_R(newbm,true);

        //输入图像到模型中
        Trace.beginSection("feed");
        inferenceInterface.feed(inputName,  ints2float(pxs,784),1, 784);
        Trace.endSection();

        //获得模型输出结果
        Trace.beginSection("run");
        String[] outputNames = new String[] {outputName};
        inferenceInterface.run(outputNames);
        Trace.endSection();

        //将输出结果存放到outputs中
        Trace.beginSection("fetch");
        inferenceInterface.fetch(outputName, outputs);
        Trace.endSection();

        //类似于tf.argmax()的功能，寻找output中最大值的index
        return argmax(outputs);
    }


}