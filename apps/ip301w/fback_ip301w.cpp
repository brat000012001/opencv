#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#include <iostream>
#include <ctime>
#include <sys/time.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <direct.h>

using namespace cv;
using namespace std;

static void help()
{
    cout <<
            "\nThis program demonstrates dense optical flow algorithm by Gunnar Farneback\n"
            "Mainly the function: calcOpticalFlowFarneback()\n"
            "Call:\n"
            "./fback_ip301w\n"
            "This reads from video camera 0\n" << endl;
}
static void drawOptFlowMap(const Mat& flow, Mat& cflowmap, int step,
                    double, const Scalar& color)
{
    for(int y = 0; y < cflowmap.rows; y += step)
        for(int x = 0; x < cflowmap.cols; x += step)
        {
            const Point2f& fxy = flow.at<Point2f>(y, x);
            line(cflowmap, Point(x,y), Point(cvRound(x+fxy.x), cvRound(y+fxy.y)),
                 color);
            circle(cflowmap, Point(x,y), 2, color, -1);
        }
}

static double computeOptFlowMapMag(const Mat& flow, Mat& cflowmap, int step)
{
    double mag = 0.0;
    for(int y = 0; y < cflowmap.rows; y += step)
        for(int x = 0; x < cflowmap.cols; x += step)
        {
            const Point2f& fxy = flow.at<Point2f>(y, x);

            mag += sqrt(fxy.dot(fxy));
        }
    return (mag * step * step) / (cflowmap.rows * cflowmap.cols);
}

class  MotionDetector;

class MotionDetectorState
{
public:
    virtual bool IsCapturing() const = 0;
    virtual void OnEnterCapture(MotionDetector &motion) = 0;
    virtual void OnFrameCaptured(MotionDetector &motionDetector, const Mat & nextFrame) = 0;
    virtual void OnLeaveCapture(MotionDetector &motion) = 0;
};

class EndOfMotionDetectionState : public MotionDetectorState
{
public:
    virtual bool IsCapturing() const;
    virtual void OnEnterCapture(MotionDetector &motion);
    virtual void OnFrameCaptured(MotionDetector &motionDetector, const Mat & nextFrame);
    virtual void OnLeaveCapture(MotionDetector &motion);
    static EndOfMotionDetectionState &defaultInstance();

};

class NoMotionDetectedState : public MotionDetectorState
{
public:
    virtual bool IsCapturing() const;
    virtual void OnEnterCapture(MotionDetector &motion);
    virtual void OnFrameCaptured(MotionDetector &motionDetector, const Mat & nextFrame);
    virtual void OnLeaveCapture(MotionDetector &motion);
    static NoMotionDetectedState &defaultInstance();
private:
    Mat m_prevgray;
};

class MotionDetectedState : public MotionDetectorState
{
public:
    MotionDetectedState();
    virtual bool IsCapturing() const;
    virtual void OnEnterCapture(MotionDetector &motion);
    virtual void OnFrameCaptured(MotionDetector &motionDetector, const Mat & nextFrame);
    virtual void OnLeaveCapture(MotionDetector &motion);
    static MotionDetectedState &defaultInstance();
private:
    VideoWriter * m_writer;
    timeval m_start;
};

class DelayMotionDetectionState : public MotionDetectorState
{
public:
    DelayMotionDetectionState();
    virtual bool IsCapturing() const;
    virtual void OnEnterCapture(MotionDetector &motion);
    virtual void OnFrameCaptured(MotionDetector &motionDetector, const Mat & nextFrame);
    virtual void OnLeaveCapture(MotionDetector &motion);
    static DelayMotionDetectionState &defaultInstance();
private:
    timeval m_start;
};

//
// MotionDetector class
//
class MotionDetector
{
public:
    MotionDetector(VideoCapture *cap, const char *outputdir) :
        m_cap(cap),
        m_state(0),
        m_outputdir(outputdir)
    {

    }

    void setState(MotionDetectorState *newState)
    {
        if (m_state)
            m_state->OnLeaveCapture(*this);
        m_state = newState;
        if (m_state)
            m_state->OnEnterCapture(*this);
    }

    void run()
    {
        help();

        Mat frame;
        namedWindow("flow", 1);

        setState(&DelayMotionDetectionState::defaultInstance());
        while(m_state->IsCapturing())
        {
            *m_cap >> frame;

            m_state->OnFrameCaptured(*this, frame);
        }
        setState(0);
    }

    VideoCapture * getCap()
    {
        return m_cap;
    }

    MotionDetectorState * getState()
    {
        return m_state;
    }
    const std::string & getOutputDir() const
    {
        return m_outputdir;
    }
private:
    VideoCapture *m_cap;
    MotionDetectorState * m_state;
    std::string m_outputdir;
};

//
bool EndOfMotionDetectionState::IsCapturing() const
{
    return false;
}
void EndOfMotionDetectionState::OnEnterCapture(MotionDetector &/*motion*/)
{
}
void EndOfMotionDetectionState::OnFrameCaptured(MotionDetector &/*motionDetector*/, const Mat & /*nextFrame*/)
{

}
void EndOfMotionDetectionState::OnLeaveCapture(MotionDetector &/*motion*/)
{

}
EndOfMotionDetectionState &EndOfMotionDetectionState::defaultInstance()
{
    static EndOfMotionDetectionState _instance;
    return _instance;
}

//
bool NoMotionDetectedState::IsCapturing() const
{
    return true;
}

void NoMotionDetectedState::OnEnterCapture(MotionDetector &/*motion*/)
{
    Mat empty;
    m_prevgray = empty;
}
void NoMotionDetectedState::OnFrameCaptured(MotionDetector &motionDetector, const Mat & nextFrame)
{
    Mat gray, flow, cflow;

    cvtColor(nextFrame, gray, COLOR_BGR2GRAY);

    double mag = 0.0;
    if( m_prevgray.data )
    {
        calcOpticalFlowFarneback(m_prevgray, gray, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
        cvtColor(m_prevgray, cflow, COLOR_GRAY2BGR);
        drawOptFlowMap(flow, cflow, 16, 1.5, Scalar(0, 255, 0));
        mag = computeOptFlowMapMag(flow, cflow, 2);

        char buf[255];
        sprintf(buf, "Mag: %f", mag);

        //putText(image, "Testing text rendering", org, rng.uniform(0,8),
        //    rng.uniform(0,100)*0.05+0.1, randomColor(rng), rng.uniform(1, 10), lineType);

        Scalar color;
        if (mag > 0.8)
            color = Scalar(0,0,255);
        else if (mag > 0.5)
            color = Scalar(0, 255, 0);
        else
            color = Scalar(255, 0, 0);
        cv::putText(cflow, buf, Point(10,80), 2, 2.5, color, 2.5, cv::LINE_AA);

        imshow("flow", cflow);

    }
    std::swap(m_prevgray, gray);
    if(waitKey(30)>=0)
    {
        motionDetector.setState(&EndOfMotionDetectionState::defaultInstance());
    }
    else if (mag > 1.8)
    {
        motionDetector.setState(&MotionDetectedState::defaultInstance());
    }
}
void NoMotionDetectedState::OnLeaveCapture(MotionDetector &/*motion*/)
{
}
NoMotionDetectedState &NoMotionDetectedState::defaultInstance()
{
    static NoMotionDetectedState _instance;
    return _instance;
}

//
MotionDetectedState::MotionDetectedState() : m_writer(0)
{

}
bool MotionDetectedState::IsCapturing() const
{
    return true;
}
void MotionDetectedState::OnEnterCapture(MotionDetector &motion)
{
    char buf[255];

    if (m_writer)
    {
        delete m_writer;
        m_writer = 0;
    }

    printf("Motion is detected: Begin capturing...\n");

    gettimeofday(&m_start, 0);

    int frame_width  =   motion.getCap()->get(CAP_PROP_FRAME_WIDTH);
    int frame_height =   motion.getCap()->get(CAP_PROP_FRAME_HEIGHT);

    int ex = static_cast<int>(VideoWriter::fourcc('X','V','I','D'));

    std::string path = motion.getOutputDir();
    path += "\\";
    path += ltoa(m_start.tv_sec, buf, 10);
    path += ".avi";

    printf("Motion is detected: Recording live video to \"%s\"...\n", path.c_str());
 
    m_writer = new VideoWriter(path.c_str(), ex, 10, Size(frame_width,frame_height),true);

    gettimeofday(&m_start, 0);
}
void MotionDetectedState::OnFrameCaptured(MotionDetector &motionDetector, const Mat & nextFrame)
{
    char buf[255];
    if (m_writer)
    {
        timeval current;
        gettimeofday(&current, 0);

        // Write the current frame
        m_writer->write(nextFrame);

        int remaining = 10 - (current.tv_sec - m_start.tv_sec);
        if (remaining > 0)
        {
            // Display the optical flow vectors and the info message to left the observers
            // know the system is currently recording 
            sprintf(buf, "Remaining %d seconds of recording", remaining);

            Mat gray;
            cvtColor(nextFrame, gray, COLOR_BGR2GRAY);
            cv::putText(gray,  buf, Point(10,80), FONT_HERSHEY_PLAIN, 1.8, Scalar(0,0,255), 1.5, cv::LINE_AA);

            imshow("flow", gray);
         }

        if (current.tv_sec - m_start.tv_sec > 10)
        {
            motionDetector.setState(&DelayMotionDetectionState::defaultInstance());
        }
        if(waitKey(30)>=0)
        {
            motionDetector.setState(&EndOfMotionDetectionState::defaultInstance());
        }
     }
}
void MotionDetectedState::OnLeaveCapture(MotionDetector &/*motion*/)
{
    printf("Motion is detected: End capturing...\n");
    //m_writer->release();
    delete m_writer;
    m_writer = 0;
}

MotionDetectedState &MotionDetectedState::defaultInstance()
{
    static MotionDetectedState _instance;
    return _instance;
}

//
DelayMotionDetectionState::DelayMotionDetectionState()
{
    m_start.tv_sec = 0;
}
bool DelayMotionDetectionState::IsCapturing() const
{
    return true;
}
void DelayMotionDetectionState::OnEnterCapture(MotionDetector &/*motion*/)
{
    gettimeofday(&m_start, 0);
}
void DelayMotionDetectionState::OnFrameCaptured(MotionDetector &motionDetector, const Mat & nextFrame)
{
    timeval current;
    char buf [255];
    gettimeofday(&current, 0);

    int remaining = 5 - (current.tv_sec - m_start.tv_sec);
    if (remaining >= 0)
    {
        sprintf(buf, "Motion detection is suspended for %d seconds", remaining);

        //Mat gray;
        //cvtColor(nextFrame, gray, COLOR_BGR2GRAY);
        cv::putText(nextFrame,  buf, Point(10,80), 2, 0.5, Scalar(0,0,255), 2.5, cv::LINE_AA);

        imshow("flow", nextFrame);
    }

    if (current.tv_sec - m_start.tv_sec > 5)
    {
        motionDetector.setState(&NoMotionDetectedState::defaultInstance());
    }
    if(waitKey(30)>=0)
    {
        motionDetector.setState(&EndOfMotionDetectionState::defaultInstance());
    }
}
void DelayMotionDetectionState::OnLeaveCapture(MotionDetector &/*motion*/)
{
}
DelayMotionDetectionState &DelayMotionDetectionState::defaultInstance()
{
    static DelayMotionDetectionState _instance;
    return _instance;
}

int main(int argc, char** argv)
{
    int camid = 0;
    VideoCapture cap;
    const char *outputdir = "data";
    struct stat info;

    if (argc > 1)
    {
        camid = atoi(argv[1]);
    }
    if (argc > 2)
    {
        outputdir = argv[2];
    }
    if (!cap.open(camid))
    {
        help();
        return -1;
    }

    if (stat(outputdir, &info) != 0)
    {
	if (_mkdir(outputdir) != 0)
	{
		printf("\"%s\" output directory does not exist and/or cannot be created.\n", outputdir);
		help();
	}
    }
    else if ((info.st_mode & _S_IFDIR) == 0) 
    {
	printf("\"%s\" is not a directory.\n", outputdir);
	help();
    }

    MotionDetector detector(&cap, outputdir);
    try
    {
        detector.run();
    }
    catch(std::exception &ex)
    {
        std::cout << ex.what() << std::endl;
    }
    return 0;
}
