#ifndef __OPENCL_TIME_H
#define __OPENCL_TIME_H
#include <iostream>
#include <chrono>
#include <string>
#ifdef __ANDROID__
#include <android/log.h>
#endif
namespace pcl
{
	/** \brief Simple stopwatch.
	* \ingroup common
	*/
	class StopWatch
	{
	public:
		/** \brief Constructor. */
		StopWatch() : start_time_(std::chrono::system_clock::now())
		{
		}

		/** \brief Destructor. */
		virtual ~StopWatch() {}

		/** \brief Retrieve the time in milliseconds spent since the last call to \a reset(). */
		inline double getTime()
		{
			std::chrono::time_point<std::chrono::system_clock> end_time = std::chrono::system_clock::now();
			std::chrono::duration<double> elapsed_seconds = end_time - start_time_;
			return elapsed_seconds.count() * 1000.0;
		}

		/** \brief Retrieve the time in seconds spent since the last call to \a reset(). */
		inline double getTimeSeconds()
		{
			std::chrono::time_point<std::chrono::system_clock> end_time = std::chrono::system_clock::now();
			std::chrono::duration<double> elapsed_seconds = end_time - start_time_;
			return elapsed_seconds.count();
		}

		/** \brief Reset the stopwatch to 0. */
		inline void reset()
		{
			start_time_ = std::chrono::system_clock::now();
		}

	protected:
		std::chrono::time_point<std::chrono::system_clock> start_time_;
	};

	/** \brief Class to measure the time spent in a scope
	*
	* To use this class, e.g. to measure the time spent in a function,
	* just create an instance at the beginning of the function. Example:
	*
	* \code
	* {
	*   pcl::ScopeTime t1 ("calculation");
	*
	*   // ... perform calculation here
	* }
	* \endcode
	*
	* \ingroup common
	*/
	class ScopeTime : public StopWatch
	{
	public:
		inline ScopeTime(const char* title) :
			title_(std::string(title))
		{
			start_time_ = std::chrono::system_clock::now();
		}

		inline ScopeTime() :
			title_(std::string(""))
		{
			start_time_ = std::chrono::system_clock::now();
		}

		inline ~ScopeTime()
		{
			double val = this->getTime();
#ifdef ANDROID	
			__android_log_print(ANDROID_LOG_DEBUG, __func__, "%s took %lf ms.", title_.c_str(), val);
#else
			std::cout << title_ << " took " << val << "ms.\n";
#endif
		}

	private:
		std::string title_;
	};
	/////////////////////////////////////////////////////////////////////////////
	struct SampledScopeTime : public StopWatch
	{
		enum { EACH = 149 };
		SampledScopeTime(double& time_ms) : time_ms_(time_ms) {}
		~SampledScopeTime()
		{
			static int i_ = 0;
			static std::chrono::time_point<std::chrono::system_clock> starttime_ = std::chrono::system_clock::now();
			time_ms_ += getTime();
			if (i_ % EACH == 0 && i_)
			{
				std::chrono::time_point<std::chrono::system_clock> endtime_ = std::chrono::system_clock::now();
				std::chrono::duration<double> elapsed_seconds = endtime_ - starttime_;
#ifdef ANDROID
				__android_log_print(ANDROID_LOG_DEBUG, __func__, "Average frame time =  %0.2f ms ( %0.2f fps )( real: %0.2f fps )", time_ms_ / EACH, 1000. * EACH / time_ms_, 1000. * (double)EACH / (elapsed_seconds.count()*1000.0));
#else
				std::cout << "Average frame time = " << time_ms_ / EACH << "ms ( " << 1000. * EACH / time_ms_ << "fps )" << "( real: " << 1000. * EACH / (elapsed_seconds.count()*1000.) << "fps )" << std::endl;
#endif
				time_ms_ = 0;
				starttime_ = endtime_;
			}
			++i_;
		}
	private:
		double& time_ms_;
	};

}
#endif //_OPENCL_TIME_H