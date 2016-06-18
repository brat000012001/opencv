#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#include <iostream>
#include <ctime>
#include <time.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <sys/types.h>
#ifdef WIN32
	#include <direct.h>
#endif
#include <string>
#include <algorithm>
#include <memory>
#include <cassert>
#include <stdexcept>
#include <future>
#include <chrono>

using namespace cv;
using namespace std;

static void help()
{
    cout <<
            "\nThis program demonstrates dense optical flow algorithm by Gunnar Farneback\n"
            "Mainly the function: calcOpticalFlowFarneback()\n"
            "Call:\n"
            "./fback_ip301w\n"
            "   This reads from video camera 0\n"
            "./fback_ip301w http://192.168.1.30/goform/video2\n"
            "   This reads images from the specified URI.\n"
            "./fback_ip301w 0 0.1\n"
            "   This read from camera 0 and configures motion detection threshold. The lower the threshold, the more sensitive the motion detection will be.\n"
            << endl;
}

class FrameProvider
{
public:
	FrameProvider() {}
	virtual ~FrameProvider() {}
	virtual FrameProvider& operator>>(Mat &buffer) = 0;
	virtual int get(int propertyId) const = 0;
};

class VideoCaptureAdapter : public FrameProvider
{
public:
	VideoCaptureAdapter(VideoCapture * cap) :
		m_cap(cap)
	{
	}

	virtual FrameProvider & operator>>(Mat &buffer)
	{
		*m_cap >> buffer;
		return *this;
	}

	virtual int get(int propertyId) const
	{
		return m_cap->get(propertyId);
	}
private:
	VideoCapture *m_cap;
};

// Reads individual images from URI
class ImageGrabberFromURI : public FrameProvider
{
public:
	ImageGrabberFromURI(const std::string &filename) :
		m_fn(filename)
	{
	}

	virtual FrameProvider & operator>>(Mat &buffer)
	{
		
		m_vc.reset(new VideoCapture());

		auto handle = async(std::launch::async, [this] { return m_vc->open(m_fn); });

		auto status = handle.wait_for(std::chrono::seconds(10));

		if (status == future_status::ready && handle.get())
		//if (m_vc->open(m_fn))
		{
			*m_vc >> m_prevBuffer;
		}
		buffer = m_prevBuffer;
		
		return *this;
	}

	virtual int get(int propertyId) const
	{
		assert(m_vc != nullptr);
		if (m_vc != nullptr)
			return m_vc->get(propertyId);
		return 0;
	}
private:
	std::string m_fn;
	std::unique_ptr<VideoCapture> m_vc;
	Mat m_prevBuffer;
};

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

    enum class ThresholdLevel : unsigned int
    {
	COLD      = 0,
	LUKEWARM,
	WARM,
	HOTTER,
	HOT,
	VERYHOT,
	MAGMA
    };

    MotionDetector(FrameProvider *cap, const char *outputdir, double threshold) :
        m_cap(cap),
        m_state(0),
        m_outputdir(outputdir),
	m_threshold(threshold)
    {
	assert(m_threshold != 0.0);
	if (m_threshold == 0.0)
		throw invalid_argument("Threshold must be greater than 0");

    }

    double getThreshold() const 
    {
	return m_threshold;
    }

    ThresholdLevel getThresholdLevel(double magnitude) const 
    {

	double dy = m_threshold / 6;
	double t = dy;
	auto level = ThresholdLevel::COLD;
	if (magnitude <= t)
		return level;

	level = ThresholdLevel::LUKEWARM;
	t += dy;

	if (magnitude <= t)
		return level;

	level = ThresholdLevel::WARM;
	t += dy;

	if (magnitude <= t)
		return level;

	level = ThresholdLevel::HOTTER;
	t += dy;

	if (magnitude <= t)
		return level;

	level = ThresholdLevel::HOT;
	t += dy;

	if (magnitude <= t)
		return level;

	level = ThresholdLevel::VERYHOT;
	t += dy;

	if (magnitude <= t)
		return level;

	return ThresholdLevel::MAGMA;
    }

    Scalar TranslateThresholdLevel2Color(ThresholdLevel level) const
    {
        Scalar color;
	switch(level)
	{
		case MotionDetector::ThresholdLevel::LUKEWARM: color = Scalar(255, 0, 0); break;
		case MotionDetector::ThresholdLevel::WARM: color = Scalar(0, 128, 0); break;
		case MotionDetector::ThresholdLevel::HOTTER: color = Scalar(0, 0, 64); break;
		case MotionDetector::ThresholdLevel::HOT: color = Scalar(0, 0, 128); break;
		case MotionDetector::ThresholdLevel::VERYHOT: color = Scalar(0, 0, 255); break;
		case MotionDetector::ThresholdLevel::MAGMA: color = Scalar(255, 0, 255); break;

		default:
		case MotionDetector::ThresholdLevel::COLD: 
			color = Scalar(128, 0, 0); 
			break;
	}
	return color;
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

    FrameProvider * getCap()
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

    void DrawOptFlowMap(const Mat& flow, Mat& cflowmap, int step, double radius) const
    {
	for(int y = 0; y < cflowmap.rows; y += step)
		for(int x = 0; x < cflowmap.cols; x += step)
		{
			const Point2f& fxy = flow.at<Point2f>(y, x);

			auto mag = sqrt(fxy.dot(fxy));
			auto level = getThresholdLevel(mag);
			auto color = TranslateThresholdLevel2Color(level);

			line(cflowmap, Point(x,y), Point(cvRound(x+fxy.x), cvRound(y+fxy.y)), color);
			circle(cflowmap, Point(x,y), radius, color, -1);
		}
    }

   double ComputeOptFlowMapMag(const Mat& flow, Mat& cflowmap, int step) const
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

    
private:
    FrameProvider *m_cap;
    MotionDetectorState * m_state;
    std::string m_outputdir;
    double m_threshold;
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

        motionDetector.DrawOptFlowMap(flow, cflow, 16, 1.5);

        mag = motionDetector.ComputeOptFlowMapMag(flow, cflow, 1);

        char buf[255];
        sprintf(buf, "Mag: %f", mag);

        //putText(image, "Testing text rendering", org, rng.uniform(0,8),
        //    rng.uniform(0,100)*0.05+0.1, randomColor(rng), rng.uniform(1, 10), lineType);

        Scalar color;

	auto level = motionDetector.getThresholdLevel(mag);

	switch(level)
	{
		case MotionDetector::ThresholdLevel::COLD: color = Scalar(128, 0, 0); break;
		case MotionDetector::ThresholdLevel::LUKEWARM: color = Scalar(255, 0, 0); break;
		case MotionDetector::ThresholdLevel::WARM: color = Scalar(0, 128, 0); break;
		case MotionDetector::ThresholdLevel::HOTTER: color = Scalar(0, 0, 64); break;
		case MotionDetector::ThresholdLevel::HOT: color = Scalar(0, 0, 128); break;
		case MotionDetector::ThresholdLevel::VERYHOT: color = Scalar(0, 0, 255); break;
		case MotionDetector::ThresholdLevel::MAGMA: color = Scalar(255, 0, 255); break;
	}
        cv::putText(cflow,  buf, Point(10,80), FONT_HERSHEY_PLAIN, 1.8, color, 1.5, cv::LINE_AA);
        //cv::putText(cflow, buf, Point(10,80), 2, 2.5, color, 2.5, cv::LINE_AA);

        imshow("flow", cflow);

    }
    std::swap(m_prevgray, gray);
    if(waitKey(30)>=0)
    {
        motionDetector.setState(&EndOfMotionDetectionState::defaultInstance());
    }
    else if (mag > motionDetector.getThreshold())
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
#ifdef WIN32
    path += "\\";
#else
    path += "/";
#endif

    time_t timer;
    struct tm* tm_info;

    time(&timer);
    tm_info = localtime(&timer);

    strftime(buf, sizeof(buf), "%Y:%m:%d %H:%M:%S", tm_info);
    //sprintf(buf, "%ld", m_start.tv_sec);

    path +=  buf;
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
    std::unique_ptr<FrameProvider> fp;
    const char *outputdir = "data";
    struct stat info;
    bool is_camera = true;
    std::string filename;

    double threshold = 1.8;
    if (argc > 1)
    {
	filename = argv[1];
	auto it = std::find_if(filename.begin(), filename.end(), [](char c) { return !std::isdigit(c);});
	is_camera = it == filename.end();
	if (is_camera)
	{
        	camid = atoi(argv[1]);
	}
    }
    if (argc > 2)
    {
        outputdir = argv[2];
    }

    if (argc > 3)
    {
	threshold = std::stod(argv[3]);
    }
    if (is_camera)
    {
    	fp.reset(new VideoCaptureAdapter(&cap));
        if (!cap.open(camid))
        {
            help();
            return -1;
        }
    }
    else
    {
	fp.reset(new ImageGrabberFromURI(filename));
        if (!cap.open(filename))
        {
	    std::cerr << "\"" << filename << "\" cannot be opened." << std::endl;
            help();
            return -1;
        }
    }
    if (stat(outputdir, &info) != 0)
    {
#ifdef WIN32
	if (_mkdir(outputdir) != 0)
	{
#else
        if (mkdir(outputdir, S_IRUSR | S_IWUSR) != 0)
        {
#endif
		cerr << "\"" << outputdir << "\" output directory does not exist and/or cannot be created." << std::endl;
		help();
		return -1;
	}
    }
#ifdef WIN32
    else if ((info.st_mode & _S_IFDIR) == 0) 
#else
    else if ((info.st_mode & S_IFDIR) == 0)
#endif
    {
	cerr << "\"" << outputdir << "\" is not a directory." << std::endl;
	help();
    }

    MotionDetector detector(fp.get(), outputdir, threshold);
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
