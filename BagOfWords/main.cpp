#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <algorithm>

using namespace cv;

/* Helper class declaration and definition */
class Caltech101
{
public:
	Caltech101::Caltech101(string datasetPath, const int numTrainingImages, const int numTestImages)
	{	
		successfullyLoaded = false;
		std::cout << "Loading Caltech 101 dataset" << std::endl;
		numImagesPerCategory = numTrainingImages + numTestImages;

		// load "Categories.txt"
		std::ifstream infile(datasetPath + "/" + "Categories.txt");
		std::cout << "\tChecking Categories.txt" << std::endl;
		if (!infile.is_open())
		{
			std::cout << "\t\tError: Cannot find Categories.txt in " << datasetPath << std::endl;
			return;
		}
		std::cout << "\t\tOK!" << std::endl;

		// Parse category names
		std::cout << "\tParsing category names" << std::endl;
		string catname;
		while (getline(infile, catname))
		{
			categoryNames.push_back(catname);
		}
		std::cout << "\t\tdone!" << std::endl;

		// set num categories
		int numCategories = (int)categoryNames.size();

		// initialize outputs size
		trainingImages = vector<vector<Mat>>(numCategories);
		trainingAnnotations = vector<vector<Rect>>(numCategories);
		testImages = vector<vector<Mat>>(numCategories);
		testAnnotations = vector<vector<Rect>>(numCategories);

		// generate training and testing indices
		randomShuffle();		

		// Load data
		std::cout << "\tLoading images and annotation files" << std::endl;
		string imgDir = datasetPath + "/" + "Images";
		string annotationDir = datasetPath + "/" + "Annotations";
		for (unsigned int catIdx = 0; catIdx < categoryNames.size(); catIdx++)
		//for (int catIdx = 0; catIdx < 1; catIdx++)
		{
			string imgCatDir = imgDir + "/" + categoryNames[catIdx];
			string annotationCatDir = annotationDir + "/" + categoryNames[catIdx];
			for (int fileIdx = 0; fileIdx < numImagesPerCategory; fileIdx++)
			{
				// use shuffled training and testing indices
				int shuffledFileIdx = indices[fileIdx];
				// generate file names
				std::stringstream imgFilename, annotationFilename;
				imgFilename << "image_" << std::setfill('0') << std::setw(4) << shuffledFileIdx << ".jpg";
				annotationFilename << "annotation_" << std::setfill('0') << std::setw(4) << shuffledFileIdx << ".txt";

				// Load image
				string imgAddress = imgCatDir + '/' + imgFilename.str();
				Mat img = imread(imgAddress, CV_LOAD_IMAGE_COLOR);
				// check image data
				if (!img.data)
				{
					std::cout << "\t\tError loading image in " << imgAddress << std::endl;
					return;
				}

				// Load annotation
				string annotationAddress = annotationCatDir + '/' + annotationFilename.str();
				std::ifstream annotationIFstream(annotationAddress);
				// Checking annotation file
				if (!annotationIFstream.is_open())
				{
					std::cout << "\t\tError: Error loading annotation in " << annotationAddress << std::endl;
					return;
				}
				int tl_col, tl_row, width, height;
				Rect annotRect;
				while (annotationIFstream >> tl_col >> tl_row >> width >> height)
				{
					annotRect = Rect(tl_col - 1, tl_row - 1, width, height);					
				}

				// Split training and testing data
				if (fileIdx < numTrainingImages)
				{
					// Training data
					trainingImages[catIdx].push_back(img);
					trainingAnnotations[catIdx].push_back(annotRect);
				}
				else
				{
					// Testing data
					testImages[catIdx].push_back(img);
					testAnnotations[catIdx].push_back(annotRect);
				}				
			}			
		}
		std::cout << "\t\tdone!" << std::endl;		
		successfullyLoaded = true;
		std::cout << "Dataset successfully loaded: " << numCategories << " categories, " << numImagesPerCategory  << " images per category" << std::endl << std::endl;
	}

	bool isSuccessfullyLoaded()	{  return successfullyLoaded; }

	void dispTrainingImage(int categoryIdx, int imageIdx)
	{		
		Mat image = trainingImages[categoryIdx][imageIdx];
		Rect annotation = trainingAnnotations[categoryIdx][imageIdx];
		rectangle(image, annotation, Scalar(255, 0, 255), 2);
		imshow("Annotated training image", image);
		waitKey(0);
		destroyWindow("Annotated training image");
	}
	
	void dispTestImage(int categoryIdx, int imageIdx)
	{
		Mat image = testImages[categoryIdx][imageIdx];
		Rect annotation = testAnnotations[categoryIdx][imageIdx];
		rectangle(image, annotation, Scalar(255, 0, 255), 2);
		imshow("Annotated test image", image);
		waitKey(0);
		destroyWindow("Annotated test image");
	}

	vector<string> categoryNames; 
	vector<vector<Mat>> trainingImages;
	vector<vector<Rect>> trainingAnnotations;
	vector<vector<Mat>> testImages;
	vector<vector<Rect>> testAnnotations;

private:
	bool successfullyLoaded;
	int numImagesPerCategory;
	vector<int> indices;
	void randomShuffle()
	{
		// set init values
		for (int i = 1; i <= numImagesPerCategory; i++) indices.push_back(i);

		// permute using built-in random generator
		random_shuffle(indices.begin(), indices.end());		
	}
};

/* Function prototypes */
void Train(const Caltech101 &Dataset, Mat &codeBook, vector<vector<Mat>> &imageDescriptors, const int numCodewords);
void Test(const Caltech101 &Dataset, const Mat codeBook, vector<vector<Mat>> const& imageDescriptors, int num);

void main(void)
{
	/* Initialize OpenCV nonfree module */
	initModule_nonfree();

	/* Put the full path of the Caltech 101 folder here */
	const string datasetPath = "C:/Users/skanet1/vision/BagOfWords/dataset/Caltech 101";

	/* Set the number of training and testing images per category */
	const int numTrainingData = 40;
	const int numTestingData = 2;

	/* Set the number of codewords*/
	const int numCodewords = 50; 

	/* Load the dataset by instantiating the helper class */
	Caltech101 Dataset(datasetPath, numTrainingData, numTestingData);

	/* Terminate if dataset is not successfull loaded */
	if (!Dataset.isSuccessfullyLoaded())
	{
		std::cout << "An error occurred, press Enter to exit" << std::endl;
		getchar();
		return;
	}	
	
	/* Variable definition */
	Mat codeBook;	
	vector<vector<Mat>> imageDescriptors;

	/* Training */
	std::cout << "Training" << std::endl;
	Train(Dataset, codeBook, imageDescriptors, numCodewords);

	/* Testing */
	std::cout << "Testing" << std::endl;
	Test(Dataset, codeBook, imageDescriptors, numCodewords);
}

/* Train BoW */
void Train(const Caltech101 &Dataset, Mat &codeBook, vector<vector<Mat>> &imageDescriptors, const int numCodewords)
{
	Ptr<FeatureDetector> detector = new SiftFeatureDetector;
	Ptr<DescriptorExtractor> extractor = new SiftDescriptorExtractor;
	Ptr<DescriptorMatcher> matcher = new BFMatcher;
	Ptr<BOWImgDescriptorExtractor> descriptor_extractor = new ::BOWImgDescriptorExtractor(extractor, matcher);

	BOWKMeansTrainer trainer(numCodewords);
	vector<cv::KeyPoint> keypoints;
	vector<vector<vector<KeyPoint>>> imageKeypoints;

	Mat D;
	imageDescriptors.resize(Dataset.trainingImages.size());

	imageKeypoints.resize(Dataset.trainingImages.size());
	for (unsigned int cat = 0; cat < Dataset.trainingImages.size(); cat++) {
		imageDescriptors[cat].resize(Dataset.trainingImages[cat].size());
		imageKeypoints[cat].resize(Dataset.trainingImages[cat].size());
		for (unsigned int im = 0; im < Dataset.trainingImages[cat].size(); im++) {
			// Get a reference to the rectangle and image
			Rect r =  Dataset.trainingAnnotations[cat][im];
			Mat image = Dataset.trainingImages[cat][im];
			Mat tmp;

			// detect keypoints
			detector->detect(image, keypoints);
			
			// filter keypoints
			keypoints.erase(
			   std::remove_if(
				  keypoints.begin(), keypoints.end(),
				  [&r](KeyPoint k){ return !r.contains(k.pt);}),
			   keypoints.end()
			);

			// compute SIFT features
			extractor->compute(image, keypoints, tmp);

			imageKeypoints[cat][im] = keypoints;
			D.push_back(tmp);
		}
	}

	std::cout << "Found Keypoints" << std::endl;

	// Add descriptors to trainer
	trainer.add(D);
	codeBook = trainer.cluster();

	std::cout << "Build Codebook" << std::endl;

	// Set Vocabulary
	descriptor_extractor->setVocabulary(codeBook);

	std::cout << "Finding Bag of Words for images" << std::endl;
	std::cout << "Testing for " << Dataset.trainingImages.size() << " Images" << std::endl;
	for (unsigned int cat = 0; cat < Dataset.trainingImages.size(); cat++) {
		for (unsigned int im = 0; im < imageDescriptors[cat].size(); im++) {
			Mat const& img = Dataset.trainingImages[cat][im];
			Mat out;
			vector<KeyPoint> &kpts = imageKeypoints[cat][im];
			descriptor_extractor->compute2(img, kpts, out);
			imageDescriptors[cat][im] = out;
		}
	}
}

/* Test BoW */
void Test(const Caltech101 &Dataset, const Mat codeBook, vector<vector<Mat>> const& imageDescriptors, int num)
{
	Ptr<FeatureDetector> detector = new SiftFeatureDetector;
	Ptr<DescriptorExtractor> extractor = new SiftDescriptorExtractor;
	Ptr<DescriptorMatcher> matcher = new BFMatcher;

	Ptr<BOWImgDescriptorExtractor> descriptor_extractor = new BOWImgDescriptorExtractor(extractor, matcher);
	descriptor_extractor->setVocabulary(codeBook);
	vector<cv::KeyPoint> keypoints;
	int total_correct = 0, total = 0;
	//std::cout << "Test size: " << Dataset.testImages.size() << std::endl;
	for (unsigned int cat = 0; cat < Dataset.testImages.size(); cat++) {
		//std::cout << "Internal Size: " << Dataset.testImages[cat].size() << std::endl;
		for (unsigned int im = 0; im < Dataset.testImages[cat].size(); im++) {
			// Get a reference to the rectangle and image
			Rect r =  Dataset.testAnnotations[cat][im];
			Mat image = Dataset.testImages[cat][im];
			Mat bag;
			// detect keypoints
			detector->detect(image, keypoints);
			
			// filter keypoints
			keypoints.erase(
			   std::remove_if(
				  keypoints.begin(), keypoints.end(),
				  [&r](KeyPoint k){ return !r.contains(k.pt);}),
			   keypoints.end()
			);

			descriptor_extractor->compute2(image, keypoints, bag);

			double min = DBL_MAX;
			int category = -1;
			for (unsigned int i = 0; i < Dataset.trainingImages.size(); i++) {
				for (unsigned int j = 0; j < Dataset.trainingImages[i].size(); j++) {
					double d = norm(bag, imageDescriptors[i][j]);
					if (d < min) {
						//std::cout << "Better Match match in category: " << i << std::endl;
						min = d;
						category = i;
					}
				}
			}

			std::ostringstream os;
			os << "test_image_" << cat << "_" << im << "_codewords_" << num << "_actual_" << Dataset.categoryNames[cat] << "_guessed_ " << Dataset.categoryNames[category] << ".jpg";
			imwrite(os.str() , image);
			//std::cout << "Best match in category: " << category << std::endl;
			if (cat == category) {
				total_correct++;
			}
			total++;
		}
	}
	std::cout << "correctly guessed " << total_correct << " out of " << total << " images" <<std::endl;
	std::cout << "rate was " << (double) total_correct / (double) total << std::endl;
	std::system("pause");
}
