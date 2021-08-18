#include <iostream>
#include <stdio.h>
#include <opencv2/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/core/core.hpp>
#include <numeric>
#include <math.h>
#include <memory>
#include <cstdint>
#include <time.h>
#include <curl/curl.h>
#include <jsoncpp/json/json.h>
#include <sstream>
#include <unistd.h>
#include "opencv2/imgproc/imgproc.hpp" 
#include "FacePreprocess.h"
#include "livefacereco.hpp"
#include "live.h"
#include "mtcnn_new.h"

#define PI 3.14159265

using namespace std;
using namespace cv;

class JadwalClass{
    public:
        bool isOn;
        string start;
        string end;
};

JadwalClass jadwal[7];
float face_thre=0.40;
float true_thre=0.60;
int input_width = 640;
int input_height = 360;
float min_face_size= input_height_def*20/100;
int pengaturan_itter = 0, pengaturan_itter_api;
int pegawai_itter = 0, pegawai_itter_api;
std::vector<std::string> list_nip;

namespace{
    std::size_t callback(char* in, std::size_t size, std::size_t num, std::string* out) {
        const std::size_t totalBytes(size * num);
        out->append(in, totalBytes);
        return totalBytes;
    }
}

double sum_score, sum_fps,sum_confidence;

Mat Zscore(const Mat &fc){
    Mat mean, std;
    meanStdDev(fc, mean, std);
    Mat fc_norm = (fc - mean) / std;
    return fc_norm;
}

inline float CosineDistance(const cv::Mat &v1, const cv::Mat &v2){
    double dot = v1.dot(v2);
    double denom_v1 = norm(v1);
    double denom_v2 = norm(v2);
    return dot / (denom_v1 * denom_v2);
}

inline double count_angle(float landmark[5][2]){
    double a = landmark[2][1] - (landmark[0][1] + landmark[1][1]) / 2;
    double b = landmark[2][0] - (landmark[0][0] + landmark[1][0]) / 2;
    double angle = atan(abs(b) / a) * 180 / PI;
    return angle;
}

inline cv::Mat draw_conclucion(String intro, double input, cv::Mat result_cnn, int position){
    char string[10];
    sprintf(string, "%.2f", input);
    std::string introString(intro);
    introString += string;
    return result_cnn;
}

void getCamParam(){
    const std::string url("http://<YOUR_HOST>/test-api-view/1/?format=json");
    CURL* curl = curl_easy_init();
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_IPRESOLVE, CURL_IPRESOLVE_V4);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 10);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    long httpCode(0);
    std::unique_ptr<std::string> httpData(new std::string());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, httpData.get());
    curl_easy_perform(curl);
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &httpCode);
    curl_easy_cleanup(curl);
    
    if (httpCode == 200){
        Json::Value jsonData;
        Json::Reader jsonReader;
    
        if (jsonReader.parse(*httpData.get(), jsonData)){
            pengaturan_itter_api = jsonData["pengaturan"].asInt();
            pegawai_itter_api = jsonData["karyawan"].asInt();
    
            if (pengaturan_itter_api == 0 && pengaturan_itter != 0) pengaturan_itter = -1;
            if (pegawai_itter_api == 0 && pegawai_itter != 0) pegawai_itter = -1;
    
            if (pengaturan_itter < pengaturan_itter_api){
                std::cout << "Parameter Updated" << pengaturan_itter<<std::endl;
    
                for(int i=0;i<=6;i++){
                    std::string isActive(jsonData["parameter"]["jadwal"][i]["isActive"].asString());
                    std::string timeFrom(jsonData["parameter"]["jadwal"][i]["timeFrom"].asString());
                    std::string timeTill(jsonData["parameter"]["jadwal"][i]["timeTill"].asString());
                    jadwal[i].isOn = (isActive == "true");
                    jadwal[i].start = timeFrom;
                    jadwal[i].end = timeTill;
                }
                float face_threshold(jsonData["parameter"]["face_threshold"].asFloat());
                face_thre = face_threshold;
                std::string true_threshold(jsonData["parameter"]["true_threshold"].asString());
                true_thre = std::stof(true_threshold);
                std::string inputwidth(jsonData["parameter"]["input_width"].asString());
                input_width = std::stof(inputwidth);
                std::string inputheight(jsonData["parameter"]["input_height"].asString());
                input_height = std::stof(inputheight);
                std::string minfacesize(jsonData["parameter"]["min_face_size"].asString());
                min_face_size= input_height_def*std::stof(minfacesize)/100;
                pengaturan_itter++;
            }
    
            if (pegawai_itter < pegawai_itter_api){
                std::cout << "Employee Updated" << std::endl;
                int itt=0;
                list_nip.clear();
    
                for(auto& td : jsonData["nip"]){
                    list_nip.push_back(jsonData["nip"][itt].asString());
                    itt++;
                }
            }
        }
    
        else{
            std::cout << "Could not parse HTTP data as JSON" << std::endl;
            std::cout << "HTTP data was:\n" << *httpData.get() << std::endl;
        }
    }

    else{
        std::cout << "Couldn't GET from " << url << " - exiting" << std::endl;
    }
}

struct MemoryStruct{
    char *memory;
	size_t size;
};

static size_t WriteMemoryCallback(void *contents, size_t size, size_t nmemb, void *userp){
	size_t realsize = size * nmemb;
	struct MemoryStruct *mem = (struct MemoryStruct *) userp;
	mem->memory = (char*)realloc(mem->memory, mem->size + realsize + 1);

	if (mem->memory == NULL){
		printf("not enough memory (realloc returned NULL)\n");
		return 0;
	}
	memcpy(&(mem->memory[mem->size]), contents, realsize);
	mem->size += realsize;
	mem->memory[mem->size] = 0;
	return realsize;
}

cv::Mat download_jpeg(char* url){
    struct MemoryStruct chunk;
	chunk.memory = (char*)malloc(1);
	chunk.size = 0;
    CURL *curl_handle;
	CURLcode res;
    curl_global_init(CURL_GLOBAL_ALL);
	curl_handle = curl_easy_init();
    curl_easy_setopt(curl_handle, CURLOPT_URL, url);
    curl_easy_setopt(curl_handle, CURLOPT_WRITEFUNCTION, WriteMemoryCallback);
    curl_easy_setopt(curl_handle, CURLOPT_WRITEDATA, (void * )&chunk);
    curl_easy_setopt(curl_handle, CURLOPT_USERAGENT, "libcurl-agent/1.0");
    res = curl_easy_perform(curl_handle);
    cv::Mat image;
    std::cout << "\tget from: " << url << std::endl;
    image = imdecode(cv::Mat(1, chunk.size, CV_8U, chunk.memory), -1);
    curl_easy_cleanup(curl_handle);
    curl_global_cleanup();
    
    if (chunk.memory) {
		free(chunk.memory);
	}
    return image;
}

int get_day(void){
    time_t t = time(NULL);
    struct tm *now = localtime(&t);
    int int_day = now->tm_wday;
    return (int_day+6)%7;
}

int compare_clock(std::string clock_compared){
    if (clock_compared == "null") return 0;
    if (clock_compared.size() ==4) clock_compared.replace(1,1, "");
    if (clock_compared.size() ==5) clock_compared.replace(2,1, "");
    time_t timetoday = time(NULL);
    struct tm *now = localtime(&timetoday);
    std::ostringstream jam;
    jam<<now->tm_hour << now->tm_min;
    std::string jam_str = jam.str();
    int clock_result = std::stoi(jam_str)-std::stoi(clock_compared);
    return clock_result;
}
int MTCNNDetection(){
    jadwal[0].isOn = false;
    jadwal[1].isOn = true;
    jadwal[2].isOn = true;
    jadwal[3].isOn = true;
    jadwal[4].isOn = true;
    jadwal[5].isOn = true;
    jadwal[6].isOn = false;
    jadwal[0].start = "null";
    jadwal[1].start = "07:00";
    jadwal[2].start = "07:00";
    jadwal[3].start = "07:00";
    jadwal[4].start = "07:00";
    jadwal[5].start = "07:00";
    jadwal[6].start = "null";
    jadwal[0].end = "null";
    jadwal[1].end = "09:00";
    jadwal[2].end = "09:00";
    jadwal[3].end = "09:00";
    jadwal[4].end = "09:00";
    jadwal[5].end = "09:00";
    jadwal[6].end = "null";

    struct ModelConfig config1 ={2.7f,0.0f,0.0f,80,80,"model_1",false};
    struct ModelConfig config2 ={4.0f,0.0f,0.0f,80,80,"model_2",false};
    vector<struct ModelConfig> configs;
    configs.emplace_back(config1);
    configs.emplace_back(config2);
    class Live live;
    live.LoadModel(configs);
    class Arcface reco;
    Mat  faces;
    vector<cv::Mat> fc1;
    std::string pattern_jpg = project_path+ "/img/*.jpg";
	std::vector<cv::String> image_names;
	cv::glob(pattern_jpg, image_names);
    int image_number=image_names.size();

	if (image_number == 0){
		std::cout << "No image files[jpg]" << std::endl;
		return 0;
	}
    cout <<"loading pictures..."<<endl;

	for (unsigned int image_ = 0; image_ < image_number; ++ image_){
		faces = cv::imread(image_names[ image_]);
        fc1.push_back(reco.getFeature(faces));
        fc1[image_] = Zscore(fc1[image_]);
        printf("\rloading[%.2lf%%]",  image_*100.0 / (image_number - 1));
    }
    cout <<""<<endl;
    cout <<"loading succeed! "<<image_number<<" pictures in total"<<endl;
    float factor = 0.709f;
    float threshold[3] = {0.7f, 0.6f, 0.6f};
    int count = 0;
    VideoCapture cap(0);
    cap.set(CAP_PROP_FRAME_WIDTH, input_width);
    cap.set(CAP_PROP_FRAME_HEIGHT, input_height);
    cap.set(CAP_PROP_FPS, 90);

    if (!cap.isOpened()){
        cerr << "cannot get image" << endl;
        return -1;
    }
    float confidence;
    vector<float> fps;
    static double current;
    static char string[10];
    static char string1[10];
    char buff[10];
    Mat frame;
    Mat result_cnn;
    float v1[5][2] = {
            {30.2946f, 51.6963f},
            {65.5318f, 51.5014f},
            {48.0252f, 71.7366f},
            {33.5493f, 92.3655f},
            {62.7299f, 92.2041f}};
    cv::Mat src(5, 2, CV_32FC1, v1);
    memcpy(src.data, v1, 2 * 5 * sizeof(float));
    double score, angle;

    while (1){
        getCamParam();
        bool today_isOn = jadwal[get_day()].isOn;
        std::string today_start = jadwal[get_day()].start;
        std::string today_end = jadwal[get_day()].end;

        if (today_isOn && compare_clock(today_start) > 0 && compare_clock(today_end) < 0){

            if (pegawai_itter < pegawai_itter_api) {

                for(std::vector<std::string>::iterator itr=list_nip.begin();itr!=list_nip.end();++itr){
                    std::string url_nip = "<YOUR_HOST>/media/faceimg/"+*itr+".jpg";
                    cv::Mat img_pegawai = download_jpeg(&url_nip[0]);
                    cv::flip (img_pegawai,img_pegawai,1);
                    vector<Bbox> faceInfoRecord = detect_mtcnn(img_pegawai, min_face_size);
                    int lagerest_face_api=0,largest_number_api=0;
                    float v2_api[5][2] =
                    {{faceInfoRecord[0].ppoint[0], faceInfoRecord[0].ppoint[5]},
                    {faceInfoRecord[0].ppoint[1], faceInfoRecord[0].ppoint[6]},
                    {faceInfoRecord[0].ppoint[2], faceInfoRecord[0].ppoint[7]},
                    {faceInfoRecord[0].ppoint[3], faceInfoRecord[0].ppoint[8]},
                    {faceInfoRecord[0].ppoint[4], faceInfoRecord[0].ppoint[9]},
                    };
                    angle = count_angle(v2_api);
                    cv::Mat dst_api(5, 2, CV_32FC1, v2_api);
                    memcpy(dst_api.data, v2_api, 2 * 5 * sizeof(float));
                    cv::Mat m_api = FacePreprocess::similarTransform(dst_api, src);
                    cv::Mat aligned_api = img_pegawai.clone();
                    cv::warpPerspective(img_pegawai, aligned_api, m_api, cv::Size(96, 112), INTER_LINEAR);
                    resize(aligned_api, aligned_api, Size(112, 112), 0, 0, INTER_LINEAR);
                    imwrite(project_path+ "/img/"+*itr+".jpg", aligned_api);
                }
                image_names.clear();
                fc1.clear();
                cv::glob(pattern_jpg, image_names);
                image_number=image_names.size();

                if (image_number == 0) {
                    std::cout << "No image files[jpg]" << std::endl;
                    return 0;
                }
                cout <<"loading pictures..."<<endl;

                for (unsigned int image_ = 0; image_ < image_number; ++ image_){
                    faces = cv::imread(image_names[ image_]);
                    fc1.push_back(reco.getFeature(faces));
                    fc1[image_] = Zscore(fc1[image_]);
                    printf("\rloading[%.2lf%%]",  image_*100.0 / (image_number - 1));
                }
                cout <<""<<endl;
                pegawai_itter++;
            }
            double t = (double) cv::getTickCount();
            cap >> frame;
            cv::flip (frame,frame,1);
            resize(frame, result_cnn, frame_size,INTER_LINEAR);
            vector<Bbox> faceInfo = detect_mtcnn(frame, min_face_size);
            int lagerest_face=0,largest_number=0;

            for (int i = 0; i < faceInfo.size(); i++){
                int y_ = (int) faceInfo[i].y2 * ratio_y;
                int h_ = (int) faceInfo[i].y1 * ratio_y;
                if (h_-y_> lagerest_face){
                    lagerest_face=h_-y_;
                    largest_number=i;
                }
            }
            int start_la,end_la;

            if (faceInfo.size()==0) {
                start_la= 0;
                end_la= 0;
            }

            else if(largest_face_only){
                start_la= largest_number;
                end_la= largest_number+1;
            }

            else {
                start_la=0;
                end_la=faceInfo.size();
            }

            for (int i =  start_la; i <end_la; i++) {
                float x_   =  faceInfo[i].x1;
                float y_   =  faceInfo[i].y1;
                float x2_ =  faceInfo[i].x2;
                float y2_ =  faceInfo[i].y2;
                int x = (int) x_ ;
                int y = (int) y_;
                int x2 = (int) x2_;
                int y2 = (int) y2_;
                struct LiveFaceBox  live_box={x_,y_,x2_,y2_} ;
                cv::rectangle(result_cnn, Point(x*ratio_x, y*ratio_y), Point(x2*ratio_x,y2*ratio_y), cv::Scalar(0, 0, 255), 2);
                float v2[5][2] =
                        {{faceInfo[i].ppoint[0], faceInfo[i].ppoint[5]},
                        {faceInfo[i].ppoint[1], faceInfo[i].ppoint[6]},
                        {faceInfo[i].ppoint[2], faceInfo[i].ppoint[7]},
                        {faceInfo[i].ppoint[3], faceInfo[i].ppoint[8]},
                        {faceInfo[i].ppoint[4], faceInfo[i].ppoint[9]},
                        };
                angle = count_angle(v2);
                static std::string hi_name;
                static std::string liveface;
                static int stranger,close_enough;
                cout<< "Detecting..."<<endl;

                if (count%jump==0){
                    cv::Mat dst(5, 2, CV_32FC1, v2);
                    memcpy(dst.data, v2, 2 * 5 * sizeof(float));
                    cv::Mat m = FacePreprocess::similarTransform(dst, src);
                    cv::Mat aligned = frame.clone();
                    cv::warpPerspective(frame, aligned, m, cv::Size(96, 112), INTER_LINEAR);
                    confidence=live.Detect(frame,live_box);
                    resize(aligned, aligned, Size(112, 112), 0, 0, INTER_LINEAR);
                    current=0;
                    int maxPosition;

                    if (confidence>true_thre){
                        cv::Mat fc2 = reco.getFeature(aligned);
                        fc2 = Zscore(fc2);
                        vector<double> score_;

                        for (unsigned int compare_ = 0; compare_ < image_number; ++ compare_){
                            score_.push_back(CosineDistance(fc1[compare_], fc2));
                        }
                        maxPosition = max_element(score_.begin(),score_.end()) - score_.begin(); 
                        current=score_[maxPosition];
                        score_.clear();
                        sprintf(string, "%.4f", current);
                    }
//                    cout<<"true: "<<confidence<<" | face: "<< current <<endl;

                    if (current >= face_thre && y2-y>= distance_threshold){
                        int slant_position=image_names[maxPosition].rfind ('/');
                        cv::String name = image_names[maxPosition].erase(0,slant_position+1);
                        name=name.erase( name.length()-4, name.length()-1);
                        hi_name="employee="+name+"&kamera=1&apikey=123abc";
                        cout<<name<<endl;
                        sprintf(string1, "%.4f", confidence);

                        liveface="Asli";
                        CURL* curl;
                        CURLcode result, res;
                        curl_global_init(CURL_GLOBAL_ALL);
                        curl = curl_easy_init();

                        if(curl) {
                            curl_easy_setopt(curl, CURLOPT_URL, "http://<YOUR_HOST>/absensi-api-view/");
                            curl_easy_setopt(curl, CURLOPT_POSTFIELDS, hi_name.c_str());
                            res = curl_easy_perform(curl);

                            if(res != CURLE_OK)
                                fprintf(stderr, "curl_easy_perform() failed: %s\n", curl_easy_strerror(res));
                            curl_easy_cleanup(curl);
                        }
                        curl_global_cleanup();
                        
                        cout<<liveface<<endl;
                        stranger=0;
                        close_enough=1;
                    } else if (current < face_thre){
                        cout<<"asing"<<endl;
                    }

                    for (int j = 0; j < 5; j += 1) {

                        if (j == 0 or j == 3) {
                            cv::circle(result_cnn, Point(faceInfo[i].ppoint[j]*ratio_x, faceInfo[i].ppoint[j + 5]*ratio_y), 3,
                                    Scalar(0, 255, 0),
                                    FILLED, LINE_AA);

                        } else if (j==2){
                            cv::circle(result_cnn, Point(faceInfo[i].ppoint[j]*ratio_x, faceInfo[i].ppoint[j + 5]*ratio_y), 3,
                                    Scalar(255, 0, 0),
                                    FILLED, LINE_AA);

                        } else {
                            cv::circle(result_cnn, Point(faceInfo[i].ppoint[j]*ratio_x, faceInfo[i].ppoint[j + 5]*ratio_y), 3,
                                    Scalar(0, 0, 255),
                                    FILLED, LINE_AA);
                        }
                    }
                }

                else{
                    if (count==10*jump-1) count=0;

                    for (int j = 0; j < 5; j += 1) {

                        if (j == 0 or j == 3) {
                            cv::circle(result_cnn, Point(faceInfo[i].ppoint[j]*ratio_x, faceInfo[i].ppoint[j + 5]*ratio_y), 3,
                                    Scalar(0, 255, 0),
                                    FILLED, LINE_AA);

                        } else if (j==2){
                            cv::circle(result_cnn, Point(faceInfo[i].ppoint[j]*ratio_x, faceInfo[i].ppoint[j + 5]*ratio_y), 3,
                                    Scalar(255, 0, 0),
                                    FILLED, LINE_AA);

                        } else {
                            cv::circle(result_cnn, Point(faceInfo[i].ppoint[j]*ratio_x, faceInfo[i].ppoint[j + 5]*ratio_y), 3,
                                    Scalar(0, 0, 255),
                                    FILLED, LINE_AA);
                        }
                    }
                }
            }
            t = ((double) cv::getTickCount() - t) / (cv::getTickFrequency());
            fps.push_back(1.0/t);
            int fpsnum_= fps.size();
            float fps_mean;

            if(fpsnum_<=30){
                sum_fps = std::accumulate(std::begin(fps), std::end(fps), 0.0);
                fps_mean = sum_fps /  fpsnum_; 
            }

            else{
                sum_fps = std::accumulate(std::end(fps)-30, std::end(fps), 0.0);
                fps_mean = sum_fps /  30; 
                if(fpsnum_>=300) fps.clear();
            }

            if (count%jump==0){
                result_cnn = draw_conclucion("FPS: ", fps_mean, result_cnn, 20);//20
                result_cnn = draw_conclucion("Angle: ", angle, result_cnn, 40);//65
            }
//            cv::imshow("image", result_cnn);
            cv::waitKey(1);
        }

        else{
            cout<<"inactive :"<<today_isOn<<" <"<<today_start<< "==" << compare_clock(today_start) <<" | "<<today_end<<"=="<<compare_clock(today_end)<<">"<<endl;
            sleep(5);
        }
    }
}
