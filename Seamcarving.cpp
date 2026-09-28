#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <limits>
using namespace cv;
using namespace std;
int clamp_value(int val, int min_val, int max_val);
int ***transpose_rgb(int ***rgb, int &H, int &W);
void remove_vertical_seam(int ***&rgb, int &H, int &W, int seamPath[][2]);
void highlightSeam(int ***rgb, int H, int seamPath[][2]);
int give_min(double l, double m, double r);
int ***alloc_rgb(int H, int W, int C);
double **alloc_double2d(int H, int W);
void free_double2d(double **m, int H);
int **alloc_int2d(int H, int W);
void free_int2d(int **m, int H);
void free_rgb(int ***a, int H, int W);
Mat rgb_to_mat(int ***rgb, int H, int W);
bool mat_to_rgb(const Mat &img, int ***&rgb, int &H, int &W, int &C, int flag = 0);
void display(int ***rgb, int H, int W);
void showSeam(int ***rgb, int H, int W, int seamPath[][2]);
void compute_min_energy_dp(double **energy, int h, int w, int **parent, double **dp);
void computeEnergy(int ***rgb, int H, int W, double **Energy);
void seamCarving(const char *in_path, const char *out_path, int targetW, int targetH);

int main(int argc, char *argv[])
{
    if (argc < 5)
    {
        cout << "Usage: " << argv[0] << " <input.jpg> <output.jpg> <new_width> <new_height>\n";
        return 1;
    }
    const char *in_path = argv[1];
    const char *out_path = argv[2];
    int targetW = atoi(argv[3]);
    int targetH = atoi(argv[4]);
    seamCarving(in_path, out_path, targetW, targetH);
    return 0;
}
int clamp_value(int val, int min_val, int max_val)
{
    if (val < min_val)
        return min_val;
    if (val > max_val)
        return max_val;
    return val;
}
int ***transpose_rgb(int ***rgb, int &H, int &W)
{
    int ***newRgb = alloc_rgb(W, H, 3);
    for (int i = 0; i < H; i++)
    {
        for (int j = 0; j < W; j++)
        {
            for (int k = 0; k < 3; k++)
            {
                newRgb[j][i][k] = rgb[i][j][k];
            }
        }
    }
    free_rgb(rgb, H, W);
    int oldH = H;
    H = W;
    W = oldH;
    return newRgb;
}
void remove_vertical_seam(int ***&rgb, int &H, int &W, int seamPath[][2])
{
    int ***newRgb = alloc_rgb(H, W - 1, 3);

    for (int i = 0; i < H; i++)
    {
        int seamCol = seamPath[i][1];
        int newJ = 0;

        for (int j = 0; j < W; j++)
        {
            if (j != seamCol)
            {
                for (int k = 0; k < 3; k++)
                {
                    newRgb[i][newJ][k] = rgb[i][j][k];
                }
                newJ++;
            }
        }
    }

    free_rgb(rgb, H, W);
    rgb = newRgb;
    W--;
}
void highlightSeam(int ***rgb, int H, int seamPath[][2])
{
    for (int i = 0; i < H; ++i)
    {
        int col = seamPath[i][1];
        rgb[i][col][0] = 255; // R
        rgb[i][col][1] = 0;   // G
        rgb[i][col][2] = 0;   // B
    }
}
int give_min(double l, double m, double r)
{
    if (l <= m && l <= r)
        return -1;
    if (m <= l && m <= r)
        return 0;
    return 1;
}
int ***alloc_rgb(int H, int W, int C)
{
    int ***a = new int **[H];
    for (int i = 0; i < H; ++i)
    {
        a[i] = new int *[W];
        for (int j = 0; j < W; ++j)
        {
            a[i][j] = new int[C];
            // not strictly needed, but zeroing is nice
            for (int k = 0; k < C; ++k)
                a[i][j][k] = 0;
        }
    }
    return a;
}
double **alloc_double2d(int H, int W)
{
    double **m = new double *[H];
    for (int i = 0; i < H; ++i)
    {
        m[i] = new double[W];
        for (int j = 0; j < W; ++j)
            m[i][j] = 0.0;
    }
    return m;
}
void free_double2d(double **m, int H)
{
    if (!m)
        return;
    for (int i = 0; i < H; ++i)
        delete[] m[i];
    delete[] m;
}
int **alloc_int2d(int H, int W)
{
    int **m = new int *[H];
    for (int i = 0; i < H; ++i)
    {
        m[i] = new int[W];
        for (int j = 0; j < W; ++j)
            m[i][j] = 0;
    }
    return m;
}
void free_int2d(int **m, int H)
{
    if (!m)
        return;
    for (int i = 0; i < H; ++i)
        delete[] m[i];
    delete[] m;
}
void free_rgb(int ***a, int H, int W)
{
    if (!a)
        return;
    for (int i = 0; i < H; ++i)
    {
        for (int j = 0; j < W; ++j)
            delete[] a[i][j];
        delete[] a[i];
    }
    delete[] a;
}
Mat rgb_to_mat(int ***rgb, int H, int W)
{
    Mat out(H, W, CV_8UC3);
    for (int i = 0; i < H; ++i)
    {
        Vec3b *rowptr = out.ptr<Vec3b>(i);
        for (int j = 0; j < W; ++j)
        {
            int R = clamp_value(rgb[i][j][0], 0, 255);
            int G = clamp_value(rgb[i][j][1], 0, 255);
            int B = clamp_value(rgb[i][j][2], 0, 255);
            rowptr[j] = Vec3b((uchar)B, (uchar)G, (uchar)R);
        }
    }
    return out;
}
bool mat_to_rgb(const Mat &img, int ***&rgb, int &H, int &W, int &C, int flag = 0)
{
    if (img.empty() || img.channels() != 3)
        return false;
    H = img.rows;
    W = img.cols;
    C = 3;
    rgb = alloc_rgb(H, W, C);
    for (int i = 0; i < H; ++i)
    {
        const Vec3b *rowptr = img.ptr<Vec3b>(i);
        for (int j = 0; j < W; ++j)
        {
            Vec3b bgr = rowptr[j];
            rgb[i][j][0] = (int)bgr[2]; // R
            rgb[i][j][1] = (int)bgr[1]; // G
            rgb[i][j][2] = (int)bgr[0]; // B
        }
    }
    return true;
}

void display(int ***rgb, int H, int W)
{
    cv::Mat imgMat = rgb_to_mat(rgb, H, W);
    const char *winName = "";
    cv::imshow(winName, imgMat);
    cv::waitKey(50); // wait 0.5s
}

void showSeam(int ***rgb, int H, int W, int seamPath[][2])
{
    highlightSeam(rgb, H, seamPath);
    display(rgb, H, W);
}
void compute_min_energy_dp(double **energy, int h, int w, int **parent, double **dp)
{
    for (int j = 0; j < w; j++)
        dp[0][j] = energy[0][j];
    for (int i = 1; i < h; i++)
    {
        for (int j = 0; j < w; j++)
        {
            double left = (j > 0) ? dp[i - 1][j - 1] : numeric_limits<double>::max();
            double middle = dp[i - 1][j];
            double right = (j < w - 1) ? dp[i - 1][(j + 1)] : numeric_limits<double>::max();

            int minVal = give_min(left, middle, right);
            dp[i][j] = energy[i][j] + dp[i - 1][j + minVal];
            parent[i][j] = j + minVal;
            // if (minVal == -1)
            //     parent[i][j] = j - 1;
            // else if (minVal == 0)
            //     parent[i][j] = j;
            // else
            //     parent[i][j] = j + 1;
        }
    }
}
void computeEnergy(int ***rgb, int H, int W, double **Energy)
{

    for (int i = 0; i < H; i++)
    {
        for (int j = 0; j < W; j++)
        {
            int jLeft = (j > 0) ? j - 1 : j;      // clamp to 0
            int jRight = (j < W - 1) ? j + 1 : j; // clamp to W-1

            int iUp = (i > 0) ? i - 1 : i;       // clamp to 0
            int iDown = (i < H - 1) ? i + 1 : i; // clamp to H-1

            double rx = fabs(rgb[i][jRight][2] - rgb[i][jLeft][2]);
            double gx = fabs(rgb[i][jRight][1] - rgb[i][jLeft][1]);
            double bx = fabs(rgb[i][jRight][0] - rgb[i][jLeft][0]);

            double ry = fabs(rgb[iDown][j][2] - rgb[iUp][j][2]);
            double gy = fabs(rgb[iDown][j][1] - rgb[iUp][j][1]);
            double by = fabs(rgb[iDown][j][0] - rgb[iUp][j][0]);

            Energy[i][j] = rx + gx + bx + ry + gy + by;
        }
    }
}

void seamCarving(const char *in_path, const char *out_path, int targetW, int targetH)
{

    cv::Mat img = cv::imread(in_path, IMREAD_COLOR);
    if (img.empty())
    {
        cout << "Error: could not open input image.\n";
        exit(1);
    }
    int H, W, C;
    int ***rgb = NULL;
    bool res = mat_to_rgb(img, rgb, H, W, C);
    if (!res)
    {

        cout << "Error: unexpected image format.\n";
        exit(1);
    }
    if (targetW < 1 || targetH < 1 || targetW > W || targetH > H)
    {
        cout << "Note: this code only supports downsizing (target <= original) and positive sizes.\n";
        cout << "Given: original " << W << "x" << H << ", target " << targetW << "x" << targetH << "\n";
        free_rgb(rgb, H, W);
        exit(1);
    }
    while (W > targetW)
    {
        double **E = alloc_double2d(H, W);
        double **dp = alloc_double2d(H, W);
        int **parent = alloc_int2d(H, W);
        computeEnergy(rgb, H, W, E);
        compute_min_energy_dp(E, H, W, parent, dp);
        double temp = dp[H - 1][0];
        int mincol = 0;
        for (int i = 1; i < W; i++)
        {
            if (dp[H - 1][i] < temp)
            {
                temp = dp[H - 1][i];
                mincol = i;
            }
        }
        int **seamPath = alloc_int2d(H, 2);

        for (int i = H - 1; i >= 0; i--)
        {
            seamPath[i][0] = i;
            seamPath[i][1] = mincol;

            mincol = parent[i][mincol];
        }

        showSeam(rgb, H, W, seamPath);
        remove_vertical_seam(rgb, H, W, seamPath);
        display(rgb, H, W);
        free_double2d(E, H);
        free_int2d(seamPath, H);
        free_double2d(dp, H);
        free_int2d(parent, H);
    }

    while (H > targetH)
    {
        rgb = transpose_rgb(rgb, H, W); // transpose
        double **E = alloc_double2d(H, W);
        double **dp = alloc_double2d(H, W);
        int **parent = alloc_int2d(H, W);

        computeEnergy(rgb, H, W, E);
        compute_min_energy_dp(E, H, W, parent, dp);

        // backtrack seam
        double temp = dp[H - 1][0];
        int mincol = 0;
        for (int i = 1; i < W; i++)
        {
            if (dp[H - 1][i] < temp)
            {
                temp = dp[H - 1][i];
                mincol = i;
            }
        }
        int **seamPath = alloc_int2d(H, 2);
        for (int i = H - 1; i >= 0; i--)
        {
            seamPath[i][0] = i;
            seamPath[i][1] = mincol;
            mincol = parent[i][mincol];
        }

        highlightSeam(rgb, H, seamPath);
        rgb = transpose_rgb(rgb, H, W);
        display(rgb, H, W);
        rgb = transpose_rgb(rgb, H, W);
        remove_vertical_seam(rgb, H, W, seamPath);
        free_int2d(seamPath, H);
        free_double2d(E, H);
        free_double2d(dp, H);
        free_int2d(parent, H);
        rgb = transpose_rgb(rgb, H, W); // transpose back

        display(rgb, H, W);
    }

    cv::Mat result = rgb_to_mat(rgb, H, W);
    free_rgb(rgb, H, W);

    cv::imshow("Final Resized Image", result);
    cv::waitKey(0);
    cv::imwrite(out_path, result);
}

// Pseudo code
// function seam_carve(image, new_width):
//     while image.width > new_width:
//         E = compute_energy(image)              # Step 1
//         M = compute_min_energy_dp(E)           # Step 2
//         seam = backtrack_min_energy_seam(M)    # Step 3
//         image = remove_seam(image, seam)       # Step 4
//     return image
