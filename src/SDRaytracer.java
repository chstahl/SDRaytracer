
import javax.swing.JFrame;
import javax.swing.JPanel;
import java.awt.Color;
import java.awt.Container;
import java.awt.BorderLayout;
import java.awt.Graphics;
import java.awt.Dimension;
import java.util.List;
import java.util.ArrayList;
import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/* Implementation of a very simple Raytracer
   Stephan Diehl, Universit�t Trier, 2010-2016
*/


public class SDRaytracer {
    int width = 1000;
    int height = 1000;

    private Future[] futureList = new Future[width];
    private int nrOfProcessors = Runtime.getRuntime().availableProcessors();
    private ExecutorService eservice = Executors.newFixedThreadPool(nrOfProcessors);

    private int maxRec = 3;
    int rayPerPixel = 1;
    int startX;
    int startY;
    int startZ;
    private Scene myScene;
    private int yAngleFactor;
    private int xAngleFactor;
    double tanFovX;
    double tanFovY;

    Color[][] image;

    private float fovx = (float) 0.628;
    private float fovy = (float) 0.628;

    public static void main(String[] args) {
        long start = System.currentTimeMillis();
        SDRaytracer sdr = new SDRaytracer();
        long end = System.currentTimeMillis();
        long time = end - start;
        System.out.println("time: " + time + " ms");
        System.out.println("nrprocs=" + sdr.nrOfProcessors);
    }

    void profileRenderImage() {
        long end;
        long start;
        long time;

        renderImage(); // initialisiere Datenstrukturen, erster Lauf verf�lscht sonst Messungen

        for (int procs = 1; procs < 6; procs++) {

            maxRec = procs - 1;
            System.out.print(procs);
            for (int i = 0; i < 10; i++) {
                start = System.currentTimeMillis();

                renderImage();

                end = System.currentTimeMillis();
                time = end - start;
                System.out.print(";" + time);
            }
            System.out.println("");
        }
    }

    SDRaytracer() {
        this(false);
    }

    SDRaytracer(boolean profiling) {
        myScene = new Scene();
        xAngleFactor = -4;
        yAngleFactor = 4;
        myScene.applyCamera(xAngleFactor, yAngleFactor);
        JFrame myFrame = new JFrame();
        if (!profiling) renderImage();
        else profileRenderImage();

        myFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        Container contentPane = myFrame.getContentPane();
        contentPane.setLayout(new BorderLayout());
        image = new Color[width][height];
        JPanel area = new JPanel() {
            @Override
            public void paint(Graphics g) {
                System.out.println("fovx=" + fovx + ", fovy=" + fovy + ", xangle=" + xAngleFactor + ", yangle=" + yAngleFactor);
                if (image == null) return;
                for (int i = 0; i < width; i++)
                    for (int j = 0; j < height; j++) {
                        g.setColor(image[i][j]);
                        // zeichne einzelnen Pixel
                        g.drawLine(i, height - j, i, height - j);
                    }
            }
        };

        myFrame.addKeyListener(new KeyAdapter() {
            @Override
            public void keyPressed(KeyEvent e) {
                boolean redraw = false;
                if (e.getKeyCode() == KeyEvent.VK_DOWN) {
                    xAngleFactor--;
                    redraw = true;
                }
                if (e.getKeyCode() == KeyEvent.VK_UP) {
                    xAngleFactor++;
                    redraw = true;
                }
                if (e.getKeyCode() == KeyEvent.VK_LEFT) {
                    yAngleFactor--;
                    redraw = true;
                }
                if (e.getKeyCode() == KeyEvent.VK_RIGHT) {
                    yAngleFactor++;
                    redraw = true;
                }
                if (redraw) {
                    myScene = new Scene();
                    myScene.applyCamera(xAngleFactor, yAngleFactor);
                    renderImage();
                    myFrame.repaint();
                }
            }
        });

        area.setPreferredSize(new Dimension(width, height));
        contentPane.add(area);
        myFrame.pack();
        myFrame.setVisible(true);
    }


    void renderImage() {
        tanFovX = Math.tan(fovx);
        tanFovY = Math.tan(fovy);
        for (int i = 0; i < width; i++) {
            futureList[i] = (Future) eservice.submit(new RaytraceTask(this, myScene, i, maxRec));
        }

        for (int i = 0; i < width; i++) {
            try {
                Color[] col = (Color[]) futureList[i].get();
                System.arraycopy(col, 0, image[i], 0, height);
            } catch (InterruptedException | ExecutionException ignored) {
            }
        }
    }


}

class RaytraceTask implements Callable {
    private SDRaytracer tracer;
    private Scene scene;
    private int i;
    private int maxRec;

    RaytraceTask(SDRaytracer tracer, Scene t, int ii, int maxRec) {
        this.tracer = tracer;
        scene = t;
        i = ii;
        this.maxRec = maxRec;
    }

    public Color[] call() {
        Color[] col = new Color[tracer.height];
        for (int j = 0; j < tracer.height; j++) {
            tracer.image[i][j] = new Color(0, 0, 0);
            for (int k = 0; k < tracer.rayPerPixel; k++) {
                double di = i + (Math.random() / 2 - 0.25);
                double dj = j + (Math.random() / 2 - 0.25);
                if (tracer.rayPerPixel == 1) {
                    di = i;
                    dj = j;
                }
                Ray eyeRay = new Ray(maxRec);
                eyeRay.setStart(tracer.startX, tracer.startY, tracer.startZ);   // ro
                eyeRay.setDir((float) (((0.5 + di) * tracer.tanFovX * 2.0) / tracer.width - tracer.tanFovX),
                        (float) (((0.5 + dj) * tracer.tanFovY * 2.0) / tracer.height - tracer.tanFovY),
                        1f);    // rd
                eyeRay.normalize();
                col[j] = Scene.addColors(tracer.image[i][j], eyeRay.rayTrace(scene, 0), 1.0f / tracer.rayPerPixel);
            }
        }
        return col;
    }
}

class Vec3D {
    float x;
    float y;
    float z;
    float w = 1;

    Vec3D(float xx, float yy, float zz) {
        x = xx;
        y = yy;
        z = zz;
    }

    Vec3D(float xx, float yy, float zz, float ww) {
        x = xx;
        y = yy;
        z = zz;
        w = ww;
    }

    Vec3D add(Vec3D v) {
        return new Vec3D(x + v.x, y + v.y, z + v.z);
    }

    Vec3D minus(Vec3D v) {
        return new Vec3D(x - v.x, y - v.y, z - v.z);
    }

    Vec3D mult(float a) {
        return new Vec3D(a * x, a * y, a * z);
    }

    void normalize() {
        float dist = (float) Math.sqrt((x * x) + (y * y) + (z * z));
        x = x / dist;
        y = y / dist;
        z = z / dist;
    }

    float dot(Vec3D v) {
        return x * v.x + y * v.y + z * v.z;
    }

    Vec3D cross(Vec3D v) {
        return new Vec3D(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x);
    }
}

class Triangle {
    Vec3D p1;
    Vec3D p2;
    Vec3D p3;
    Color color;
    Vec3D normal;
    float shininess;

    Triangle(Vec3D pp1, Vec3D pp2, Vec3D pp3, Color col, float sh) {
        p1 = pp1;
        p2 = pp2;
        p3 = pp3;
        color = col;
        shininess = sh;
        Vec3D e1 = p2.minus(p1),
                e2 = p3.minus(p1);
        normal = e1.cross(e2);
        normal.normalize();
    }

    void apply(Matrix m) {

        p1 = m.mult(p1);
        p2 = m.mult(p2);
        p3 = m.mult(p3);
        Vec3D e1 = p2.minus(p1),
                e2 = p3.minus(p1);
        normal = e1.cross(e2);
        normal.normalize();

    }

}


class Ray {
    Vec3D start = new Vec3D(0, 0, 0);
    Vec3D dir = new Vec3D(0, 0, 0);
    int maxRec = 1;

    Ray(int maxRec) {
        this.maxRec = maxRec;
    }

    void setStart(float x, float y, float z) {
        start = new Vec3D(x, y, z);
    }

    void setDir(float dx, float dy, float dz) {
        dir = new Vec3D(dx, dy, dz);
    }

    void normalize() {
        dir.normalize();
    }

    // see M�ller&Haines, page 305
    private IPoint intersect(Triangle t) {
        float epsilon = IPoint.EPSILON;
        Vec3D e1 = t.p2.minus(t.p1);
        Vec3D e2 = t.p3.minus(t.p1);
        Vec3D p = dir.cross(e2);
        float a = e1.dot(p);
        if ((a > -epsilon) && (a < epsilon)) return new IPoint(null, null, -1);
        float f = 1 / a;
        Vec3D s = start.minus(t.p1);
        float u = f * s.dot(p);
        if ((u < 0.0) || (u > 1.0)) return new IPoint(null, null, -1);
        Vec3D q = s.cross(e1);
        float v = f * dir.dot(q);
        if ((v < 0.0) || (u + v > 1.0)) return new IPoint(null, null, -1);
        // intersection point is u,v
        float dist = f * e2.dot(q);
        if (dist < epsilon) return new IPoint(null, null, -1);
        Vec3D ip = t.p1.mult(1 - u - v).add(t.p2.mult(u)).add(t.p3.mult(v));
        //DEBUG.debug("Intersection point: "+ip.x+","+ip.y+","+ip.z);
        return new IPoint(t, ip, dist);
    }

    IPoint hitObject(Scene scene) {
        IPoint isect = new IPoint(null, null, -1);
        float idist = -1;
        for (Triangle t : scene.getTriangles()) {
            IPoint ip = intersect(t);
            if (ip.dist != -1 && ((idist == -1) || (ip.dist < idist))) { // save that intersection
                idist = ip.dist;
                isect.iPoint = ip.iPoint;
                isect.dist = ip.dist;
                isect.triangle = t;
            }
        }
        return isect;  // return intersection point and normal
    }

    Color rayTrace(Scene s, int rec) {
        if (rec > maxRec) return Color.BLACK;
        IPoint ip = this.hitObject(s);
        if (ip.dist > IPoint.EPSILON)
            return s.lighting(this, ip, rec);
        else
            return Color.BLACK;
    }

}

class IPoint {
    static final float EPSILON = 0.0001f;
    Triangle triangle;
    Vec3D iPoint;
    float dist;

    IPoint(Triangle tt, Vec3D ip, float d) {
        triangle = tt;
        iPoint = ip;
        dist = d;
    }
}

class Light {
    Color color;
    Vec3D position;

    Light(Vec3D pos, Color c) {
        position = pos;
        color = c;
    }
}

class Scene {
    private Color ambientColor;


    private List<Triangle> triangles;


    private final Light[] lights;

    Scene() {

        ambientColor = new Color(0.01f, 0.01f, 0.01f);
        Light mainLight = new Light(new Vec3D(0, 100, 0), new Color(0.1f, 0.1f, 0.1f));
        lights = new Light[]{mainLight
                , new Light(new Vec3D(100, 200, 300), new Color(0.5f, 0, 0.0f))
                , new Light(new Vec3D(-100, 200, 300), new Color(0.0f, 0, 0.5f))
        };
        triangles = new ArrayList<>();


        addCube(0, 35, 0, 10, 10, 10, new Color(0.3f, 0, 0), 0.4f);       //rot, klein
        addCube(-70, -20, -20, 20, 100, 100, new Color(0f, 0, 0.3f), .4f);
        addCube(-30, 30, 40, 20, 20, 20, new Color(0, 0.4f, 0), 0.2f);        // gr�n, klein
        addCube(50, -20, -40, 10, 80, 100, new Color(.5f, .5f, .5f), 0.2f);
        addCube(-70, -26, -40, 130, 3, 40, new Color(.5f, .5f, .5f), 0.2f);


    }

    void applyCamera(int xAngleFactor, int yAngleFactor) {
        Matrix mRx = Matrix.createXRotation((float) (xAngleFactor * Math.PI / 16));
        Matrix mRy = Matrix.createYRotation((float) (yAngleFactor * Math.PI / 16));
        Matrix mT = Matrix.createTranslation(0, 0, 200);
        Matrix m = mT.mult(mRx).mult(mRy);
        m.print();
        for (Triangle t : triangles) {
            t.apply(m);
        }
    }

    void addCube(int x, int y, int z, int w, int h, int d, Color c, float sh) {
        triangles.add(new Triangle(new Vec3D(x, y, z), new Vec3D(x + w, y, z), new Vec3D(x, y + h, z), c, sh));
        triangles.add(new Triangle(new Vec3D(x + w, y, z), new Vec3D(x + w, y + h, z), new Vec3D(x, y + h, z), c, sh));
        //left
        triangles.add(new Triangle(new Vec3D(x, y, z + d), new Vec3D(x, y, z), new Vec3D(x, y + h, z), c, sh));
        triangles.add(new Triangle(new Vec3D(x, y + h, z), new Vec3D(x, y + h, z + d), new Vec3D(x, y, z + d), c, sh));
        //right
        triangles.add(new Triangle(new Vec3D(x + w, y, z), new Vec3D(x + w, y, z + d), new Vec3D(x + w, y + h, z), c, sh));
        triangles.add(new Triangle(new Vec3D(x + w, y + h, z), new Vec3D(x + w, y, z + d), new Vec3D(x + w, y + h, z + d), c, sh));
        //top
        triangles.add(new Triangle(new Vec3D(x + w, y + h, z), new Vec3D(x + w, y + h, z + d), new Vec3D(x, y + h, z), c, sh));
        triangles.add(new Triangle(new Vec3D(x, y + h, z), new Vec3D(x + w, y + h, z + d), new Vec3D(x, y + h, z + d), c, sh));
        //bottom
        triangles.add(new Triangle(new Vec3D(x + w, y, z), new Vec3D(x, y, z), new Vec3D(x, y, z + d), c, sh));
        triangles.add(new Triangle(new Vec3D(x, y, z + d), new Vec3D(x + w, y, z + d), new Vec3D(x + w, y, z), c, sh));
        //back
        triangles.add(new Triangle(new Vec3D(x, y, z + d), new Vec3D(x, y + h, z + d), new Vec3D(x + w, y, z + d), c, sh));
        triangles.add(new Triangle(new Vec3D(x + w, y, z + d), new Vec3D(x, y + h, z + d), new Vec3D(x + w, y + h, z + d), c, sh));
    }

    Color lighting(Ray ray, IPoint ip, int rec) {
        Vec3D point = ip.iPoint;
        Triangle triangle = ip.triangle;
        Color color = addColors(triangle.color, ambientColor, 1);
        Ray shadowRay = new Ray(ray.maxRec);
        for (Light light : lights) {
            shadowRay.start = point;
            shadowRay.dir = light.position.minus(point).mult(-1);
            shadowRay.dir.normalize();
            IPoint ip2 = shadowRay.hitObject(this);
            if (ip2.dist < IPoint.EPSILON) {
                float ratio = Math.max(0, shadowRay.dir.dot(triangle.normal));
                color = addColors(color, light.color, ratio);
            }
        }
        Ray reflection = new Ray(ray.maxRec);
        //R = 2N(N*lVect)-lVect)    lVect ausgehender Vektor
        Vec3D lVect = ray.dir.mult(-1);
        reflection.start = point;
        reflection.dir = triangle.normal.mult(2 * triangle.normal.dot(lVect)).minus(lVect);
        reflection.dir.normalize();
        Color rcolor = reflection.rayTrace(this, rec + 1);
        float ratio = (float) Math.pow(Math.max(0, reflection.dir.dot(lVect)), triangle.shininess);
        color = addColors(color, rcolor, ratio);
        return (color);
    }

    List<Triangle> getTriangles() {
        return triangles;
    }

    static Color addColors(Color c1, Color c2, float ratio) {
        return new Color(Math.min((c1.getRed() / 255f + c2.getRed() / 255f * ratio), 1f),
                Math.min((c1.getGreen() / 255f + c2.getGreen() / 255f * ratio), 1f),
                Math.min((c1.getBlue() / 255f + c2.getBlue() / 255f * ratio), 1f));
    }

}

class Matrix {
    private float[][] val = new float[4][4];

    Matrix() {
    }

    Matrix(float[][] vs) {
        val = vs;
    }

    void print() {
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                System.out.print(" " + (val[i][j] + "       ").substring(0, 8));
            }
            System.out.println();
        }
    }


    Matrix mult(Matrix m) {
        Matrix r = new Matrix();
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++) {
                float sum = 0f;
                for (int k = 0; k < 4; k++) sum = sum + val[i][k] * m.val[k][j];
                r.val[i][j] = sum;
            }
        return r;
    }

    Vec3D mult(Vec3D v) {
        Vec3D temp = new Vec3D(val[0][0] * v.x + val[0][1] * v.y + val[0][2] * v.z + val[0][3] * v.w,
                val[1][0] * v.x + val[1][1] * v.y + val[1][2] * v.z + val[1][3] * v.w,
                val[2][0] * v.x + val[2][1] * v.y + val[2][2] * v.z + val[2][3] * v.w,
                val[3][0] * v.x + val[3][1] * v.y + val[3][2] * v.z + val[3][3] * v.w);
        
        temp.x = temp.x / temp.w;
        temp.y = temp.y / temp.w;
        temp.z = temp.z / temp.w;
        temp.w = 1;
        return temp;
    }

    static Matrix createId() {
        return new Matrix(new float[][]{
                {1, 0, 0, 0},
                {0, 1, 0, 0},
                {0, 0, 1, 0},
                {0, 0, 0, 1}});
    }

    static Matrix createXRotation(float angle) {
        return new Matrix(new float[][]{
                {1, 0, 0, 0},
                {0, (float) Math.cos(angle), (float) -Math.sin(angle), 0},
                {0, (float) Math.sin(angle), (float) Math.cos(angle), 0},
                {0, 0, 0, 1}});
    }

    static Matrix createYRotation(float angle) {
        return new Matrix(new float[][]{
                {(float) Math.cos(angle), 0, (float) Math.sin(angle), 0},
                {0, 1, 0, 0},
                {(float) -Math.sin(angle), 0, (float) Math.cos(angle), 0},
                {0, 0, 0, 1}});
    }

    static Matrix createZRotation(float angle) {
        return new Matrix(new float[][]{
                {(float) Math.cos(angle), (float) -Math.sin(angle), 0, 0},
                {(float) Math.sin(angle), (float) Math.cos(angle), 0, 0},
                {0, 0, 1, 0},
                {0, 0, 0, 1}});
    }

    static Matrix createTranslation(float dx, float dy, float dz) {
        return new Matrix(new float[][]{
                {1, 0, 0, dx},
                {0, 1, 0, dy},
                {0, 0, 1, dz},
                {0, 0, 0, 1}});
    }


}


