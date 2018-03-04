package com.example.david.myapplication;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.widget.ImageView;
import android.widget.TextView;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        TF_MINIST m=new TF_MINIST(getAssets());
        Bitmap bitmap= BitmapFactory.decodeResource(getResources(),R.drawable.timg);
        TextView tv=findViewById(R.id.DOutput);
        ImageView im=findViewById(R.id.DImg);
        im.setImageBitmap(bitmap);
        tv.append("The digit is "+m.getAddResult(bitmap));
    }
}
