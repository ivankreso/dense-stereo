#include "sgm_stereo.h"
#include <stack>
#include <algorithm>
#include <stdexcept>

namespace recon {

SGMStereo::SGMStereo() : disp_range_(kDisparityRange),
                         disparity_factor_(kDisparityFactor),
                         P1_(kP1),
                         P2_(kP2),
                         consistency_threshold_(kConsistencyThreshold) {}

void SGMStereo::SetSmoothnessCostParameters(const int P1, const int P2) {
  if (P1 < 0 || P2 < 0) {
    throw std::invalid_argument("[SGMStereo::SetSmoothnessCostParameters] smoothness penalty \
                                value is less than zero");
  }
  if (P1 >= P2) {
    throw std::invalid_argument("[SGMStereo::AetSmoothnessCostParameters] small value of \
                                smoothness penalty must be smaller than large penalty value");
  }

  P1_ = static_cast<CostType>(P1);
  P2_ = static_cast<CostType>(P2);
}

void SGMStereo::SetConsistencyThreshold(const int consistency_threshold) {
  if (consistency_threshold < 0) {
    throw std::invalid_argument("[SGMStereo::setConsistencyThreshold] threshold for LR \
                                consistency must be positive");
  }
  consistency_threshold_ = consistency_threshold;
}

void SGMStereo::Compute(const std::string left_descriptors_path,
                        const std::string right_descriptors_path,
                        cv::Mat* disparity) {

  DescriptorTensor left_descriptors, right_descriptors;
  LoadRepresentationFromFile(left_descriptors_path, &left_descriptors);
  LoadRepresentationFromFile(right_descriptors_path, &right_descriptors);

  Initialize(left_descriptors, right_descriptors);

  std::cout << "Computing data costs...\n";
  ComputeCostImage(left_descriptors, right_descriptors);

  DisparityType* left_disp_image = new DisparityType[width_*height_];
  std::cout << "Computing left to right SGM...\n";
  PerformSGM(left_cost_, left_disp_image);
  DisparityType* right_disp_image = new DisparityType[width_*height_];
  std::cout << "Computing right to left SGM...\n";
  PerformSGM(right_cost_, right_disp_image);

  std::cout << "Computing disparity image...\n";
  // TODO
  EnforceLeftRightConsistency(left_disp_image, right_disp_image);

  disparity->create(height_, width_, CV_16U);
  for (int y = 0; y < height_; ++y) {
    for (int x = 0; x < width_; ++x) {
      //std::cout << left_disp_image[width_*y + x] << "\n";
      DisparityType scaled_disp = std::round(left_disp_image[width_*y + x]);
      disparity->at<uint16_t>(y,x) = static_cast<uint16_t>(scaled_disp);
    }
  }

  FreeDataBuffer();
  delete[] left_disp_image;
  delete[] right_disp_image;
}


void SGMStereo::Initialize(const DescriptorTensor& left_desc, const DescriptorTensor& right_desc) {
  SetImageSize(left_desc, right_desc);
  AllocateDataBuffer();
}

void SGMStereo::SetImageSize(const DescriptorTensor& left_desc, const DescriptorTensor& right_desc) {
  width_ = static_cast<int>(left_desc[0].size());
  height_ = static_cast<int>(left_desc.size());
  if (right_desc[0].size() != width_ || right_desc.size() != height_) {
    throw std::invalid_argument("[SGMStereo::setImageSize] sizes of left and right images are different");
  }
}

void SGMStereo::AllocateDataBuffer() {
  left_cost_ = new CostType[width_ * height_ * disp_range_]();
  right_cost_ = new CostType[width_ * height_ * disp_range_]();

  // size of the final summed costs
  int sum_cost_size = width_ * height_ * disp_range_;
  sum_cost_ = new CostType[sum_cost_size];

  // size of aggregated cost buffer for one image row
  int lr_size = width_ * disp_range_;
  for (int i = 0; i < kNumPaths; i++) {
    // buffers for storing the min values across all disparities for each path
    // which are then used to normalize the aggregated cost to achieve upper bound: L <= C_max + P2
    lr_min_prev_[i] = new CostType[width_];
    lr_min_curr_[i] = new CostType[width_];
    // buffers used to store the aggregated costs for each path and each disparity value
    lr_curr_[i] = new CostType[lr_size];
    lr_prev_[i] = new CostType[lr_size];
  }
}

void SGMStereo::FreeDataBuffer() {
  delete[] left_cost_;
  delete[] right_cost_;
  delete[] sum_cost_;
  for (int i = 0; i < kNumPaths; i++) {
    delete[] lr_min_prev_[i];
    delete[] lr_min_curr_[i];
    delete[] lr_prev_[i];
    delete[] lr_curr_[i];
  }
}

void SGMStereo::ComputeCostImage(const DescriptorTensor& left_descriptors,
                                 const DescriptorTensor& right_descriptors) {
  ComputeLeftCostImage(left_descriptors, right_descriptors);
  ComputeRightCostImage();
}


void SGMStereo::ComputeLeftCostImage(const DescriptorTensor& left_descriptors,
                                     const DescriptorTensor& right_descriptors) {
  int y_skip = width_ * disp_range_;
  #pragma omp parallel for
  for(int y = 0; y < height_; y++) {
    for(int x = 0; x < width_; x++) {
      for(int d = 0; d < disp_range_; d++) {
        int idx = y*y_skip + x*disp_range_ + d;
        if (x >= d) {
          left_cost_[idx] = (left_descriptors[y][x] - right_descriptors[y][x-d]).norm();
          assert(left_cost_[idx] >= 0 && left_cost_[idx] < 1000);
          //std::cout << left_cost_[idx] << "\n";
        }
        else {
          // TODO
          left_cost_[idx] = left_cost_[idx-1];
          assert(left_cost_[idx] >= 0 && left_cost_[idx] < 1000);
          //left_cost_[idx] = std::numeric_limits<CostType>::max();
        }
      }
    }
  }
}

void SGMStereo::ComputeRightCostImage() {
  const int widthStepCost = width_*disp_range_;

  for (int y = 0; y < height_; ++y) {
    CostType* leftCostRow = left_cost_ + widthStepCost*y;
    CostType* rightCostRow = right_cost_ + widthStepCost*y;

    for (int x = 0; x < disp_range_; ++x) {
      CostType* leftCostPointer = leftCostRow + disp_range_*x;
      CostType* rightCostPointer = rightCostRow + disp_range_*x;
      for (int d = 0; d <= x; ++d) {
        *(rightCostPointer) = *(leftCostPointer);
        rightCostPointer -= disp_range_ - 1;
        ++leftCostPointer;
      }
    }

    for (int x = disp_range_; x < width_; ++x) {
      CostType* leftCostPointer = leftCostRow + disp_range_*x;
      CostType* rightCostPointer = rightCostRow + disp_range_*x;
      for (int d = 0; d < disp_range_; ++d) {
        *(rightCostPointer) = *(leftCostPointer);
        rightCostPointer -= disp_range_ - 1;
        ++leftCostPointer;
      }
    }

    for (int x = width_ - disp_range_ + 1; x < width_; ++x) {
      int maxDisparityIndex = width_ - x;
      CostType lastValue = *(rightCostRow + disp_range_*x + maxDisparityIndex - 1);

      CostType* rightCostPointer = rightCostRow + disp_range_*x + maxDisparityIndex;
      for (int d = maxDisparityIndex; d < disp_range_; ++d) {
        *(rightCostPointer) = lastValue;
        ++rightCostPointer;
      }
    }
  }
}

void SGMStereo::PerformSGM(const CostType* data_cost, DisparityType* disparity_img) {
  const CostType kCostMax = std::numeric_limits<CostType>::max();

  // we have 2 passes each aggregating the costs from 4 paths
  // where 1 pass starts from top left pixel and 2 pass from bottom right pixel
  const int kNumPasses = 2;
  for (int pass_cnt = 0; pass_cnt < kNumPasses; pass_cnt++) {
    int startX, endX, stepX;
    int startY, endY, stepY;
    // first pass performs row-wise iteration starting from top left pixel
    if (pass_cnt == 0) {
      startX = 0; endX = width_; stepX = 1;
      startY = 0; endY = height_; stepY = 1;
    }
    // second pass performs reverse row-wise iteration starting from bottom right pixel
    else {
      startX = width_ - 1; endX = -1; stepX = -1;
      startY = height_ - 1; endY = -1; stepY = -1;
    }

    // iterate over rows
    for (int y = startY; y != endY; y += stepY) {
      // pointer to data cost for current row
      const CostType* data_cost_row = data_cost + y*width_*disp_range_;
      // pointer to aggregated cost for current row
      CostType* sum_cost_row = sum_cost_ + y*width_*disp_range_;

      // iterate over columns
      for (int x = startX; x != endX; x += stepX) {
        // pointers to 4 paths cost for current pixel
        const CostType* lr_p[kNumPaths] = { nullptr };
        // buffer to store temp minimum costs for each of the 4 paths
        CostType min_lr_p[kNumPaths] = { 0.0 };

        // set the write pointers for each path for the current pixel
        int x_skip = x * disp_range_;
        CostType* lr_curr_p[kNumPaths];
        for (int r = 0; r < kNumPaths; r++)
          lr_curr_p[r] = lr_curr_[r] + x_skip;

        // set the pointers for each path cost and min_cost
        // to appropriate neighbour of the current pixel
        if (x != startX) {
          // 1. left/right pixel is in current row lr_curr_
          lr_p[0] = lr_curr_[0] + (x-stepX)*disp_range_;
          min_lr_p[0] = lr_min_curr_[0][x-stepX];
        }
        if (y != startY) {
          // 2. upper/lower center pixel is in lr_prev_
          lr_p[2] = lr_prev_[2] + x*disp_range_;
          min_lr_p[2] = lr_min_prev_[2][x];
          // 3. upper/lower left pixel is in lr_prev_
          if (x != startX) {
            lr_p[1] = lr_prev_[1] + (x-stepX)*disp_range_;
            min_lr_p[1] = lr_min_prev_[1][x-stepX];
          }
          // 4. upper/lower right pixel is in lr_prev_
          if (x != (endX-stepX)) {
            lr_p[3] = lr_prev_[3] + (x+stepX)*disp_range_;
            min_lr_p[3] = lr_min_prev_[3][x+stepX];
          }
        }

        const CostType* dc_p = data_cost_row + x_skip;
        for (int r = 0; r < kNumPaths; r++)
          lr_min_curr_[r][x] = kCostMax;
        CostType* sum_cost_p = sum_cost_row + x_skip;
        // iterate over disparities
        for (int d = 0; d < disp_range_; d++) {
          // code below computes the following SGM cost:
          // L_r(p, d) = C(p, d) + min(L_r(p-r, d),
          // L_r(p-r, d-1) + P1, L_r(p-r, d+1) + P1,
          // min_k L_r(p-r, k) + P2) - min_k L_r(p-r, k)
          // where p = (x,y), r is one of the directions.

          // aggregate costs for each path
          for (int r = 0; r < kNumPaths; r++) {
            // we dont need to initialize lr_curr_ in the beginning because of this line
            lr_curr_p[r][d] = dc_p[d];
            // fist check if the path exists
            if (lr_p[r] != nullptr) {
              CostType agg_cost = std::min(min_lr_p[r] + P2_, lr_p[r][d]);
              // check if disp-1 exists
              if (d > 0) {
                CostType tmp_cost = lr_p[r][d-1] + P1_;
                // we need min
                if (agg_cost > tmp_cost)
                  agg_cost = tmp_cost;
              }
              // check if disp+1 exists
              if (d < (disp_range_ - 1)) {
                CostType tmp_cost = lr_p[r][d+1] + P1_;
                // we need min
                if (agg_cost > tmp_cost)
                  agg_cost = tmp_cost;
              }
              lr_curr_p[r][d] += agg_cost - min_lr_p[r];
            }
            // if the cost is smallest so far set new min which is needed for next iterations
            if (lr_min_curr_[r][x] > lr_curr_p[r][d])
              lr_min_curr_[r][x] = lr_curr_p[r][d];
            // finally sum the costs over all paths
            sum_cost_p[d] += lr_curr_p[r][d];
          }
        }
      }

      // compute the disparity map
      if (pass_cnt == kNumPasses - 1) {
        DisparityType* disparityRow = disparity_img + width_*y;

        for (int x = 0; x < width_; ++x) {
          const CostType* costSumCurrent = sum_cost_row + disp_range_*x;
          CostType bestSumCost = costSumCurrent[0];
          int bestDisparity = 0;
          for (int d = 1; d < disp_range_; ++d) {
            if (costSumCurrent[d] < bestSumCost) {
              bestSumCost = costSumCurrent[d];
              bestDisparity = d;
            }
          }
          //std::cout << bestDisparity << "\n";

          if (bestDisparity > 0 && bestDisparity < disp_range_ - 1) {
            CostType centerCostValue = costSumCurrent[bestDisparity];
            CostType leftCostValue = costSumCurrent[bestDisparity - 1];
            CostType rightCostValue = costSumCurrent[bestDisparity + 1];
            if (rightCostValue < leftCostValue) {
              disparityRow[x] = static_cast<CostType>(bestDisparity*disparity_factor_
                  + static_cast<double>(rightCostValue - leftCostValue) /
                  (centerCostValue - leftCostValue)/2.0*disparity_factor_ + 0.5);
            }
            else {
              disparityRow[x] = static_cast<CostType>(bestDisparity*disparity_factor_
                  + static_cast<double>(rightCostValue - leftCostValue) /
                  (centerCostValue - rightCostValue)/2.0*disparity_factor_ + 0.5);
            }
          }
          else {
           disparityRow[x] = static_cast<CostType>(bestDisparity*disparity_factor_);
          }
        }
      }

      // set current row buffer as previous row buffer for next iteration
      std::swap(lr_curr_, lr_prev_);
      std::swap(lr_min_curr_, lr_min_prev_);
    }
  }

  // TODO check this code
  SpeckleFilter(100, static_cast<int>(2*disparity_factor_), disparity_img);
}

void SGMStereo::SpeckleFilter(const int maxSpeckleSize, const int maxDifference, DisparityType* image) const {
  std::vector<int> labels(width_*height_, 0);
  std::vector<bool> regionTypes(1);
  regionTypes[0] = false;

  int currentLabelIndex = 0;

  for (int y = 0; y < height_; ++y) {
    for (int x = 0; x < width_; ++x) {
      int pixelIndex = width_*y + x;
      if (image[width_*y + x] != 0) {
        if (labels[pixelIndex] > 0) {
          if (regionTypes[labels[pixelIndex]]) {
            image[width_*y + x] = 0;
          }
        } else {
          std::stack<int> wavefrontIndices;
          wavefrontIndices.push(pixelIndex);
          ++currentLabelIndex;
          regionTypes.push_back(false);
          int regionPixelTotal = 0;
          labels[pixelIndex] = currentLabelIndex;

          while (!wavefrontIndices.empty()) {
            int currentPixelIndex = wavefrontIndices.top();
            wavefrontIndices.pop();
            int currentX = currentPixelIndex%width_;
            int currentY = currentPixelIndex/width_;
            ++regionPixelTotal;
            CostType pixelValue = image[width_*currentY + currentX];

            if (currentX < width_ - 1 && labels[currentPixelIndex + 1] == 0
                && image[width_*currentY + currentX + 1] != 0
                && std::abs(pixelValue - image[width_*currentY + currentX + 1]) <= maxDifference)
            {
              labels[currentPixelIndex + 1] = currentLabelIndex;
              wavefrontIndices.push(currentPixelIndex + 1);
            }

            if (currentX > 0 && labels[currentPixelIndex - 1] == 0
                && image[width_*currentY + currentX - 1] != 0
                && std::abs(pixelValue - image[width_*currentY + currentX - 1]) <= maxDifference)
            {
              labels[currentPixelIndex - 1] = currentLabelIndex;
              wavefrontIndices.push(currentPixelIndex - 1);
            }

            if (currentY < height_ - 1 && labels[currentPixelIndex + width_] == 0
                && image[width_*(currentY + 1) + currentX] != 0
                && std::abs(pixelValue - image[width_*(currentY + 1) + currentX]) <= maxDifference)
            {
              labels[currentPixelIndex + width_] = currentLabelIndex;
              wavefrontIndices.push(currentPixelIndex + width_);
            }

            if (currentY > 0 && labels[currentPixelIndex - width_] == 0
                && image[width_*(currentY - 1) + currentX] != 0
                && std::abs(pixelValue - image[width_*(currentY - 1) + currentX]) <= maxDifference)
            {
              labels[currentPixelIndex - width_] = currentLabelIndex;
              wavefrontIndices.push(currentPixelIndex - width_);
            }
          }

          if (regionPixelTotal <= maxSpeckleSize) {
            regionTypes[currentLabelIndex] = true;
            image[width_*y + x] = 0;
          }
        }
      }
    }
  }
}

void SGMStereo::EnforceLeftRightConsistency(DisparityType* left_disparity_image,
                                            DisparityType* right_disparity_image) const {
  // Check left disparity image
  for (int y = 0; y < height_; ++y) {
    for (int x = 0; x < width_; ++x) {
      if (left_disparity_image[width_*y + x] == 0) continue;

      int leftDisparityValue = static_cast<int>(static_cast<double>(
            left_disparity_image[width_*y + x])/disparity_factor_ + 0.5);
      if (x - leftDisparityValue < 0) {
        left_disparity_image[width_*y + x] = 0;
        continue;
      }

      int rightDisparityValue = static_cast<int>(static_cast<double>(
            right_disparity_image[width_*y + x-leftDisparityValue])/disparity_factor_ + 0.5);
      if (rightDisparityValue == 0 || abs(leftDisparityValue - rightDisparityValue) > consistency_threshold_) {
        left_disparity_image[width_*y + x] = 0;
      }
    }
  }

  // Check right disparity image
  for (int y = 0; y < height_; ++y) {
    for (int x = 0; x < width_; ++x) {
      if (right_disparity_image[width_*y + x] == 0)  continue;

      int rightDisparityValue = static_cast<int>(static_cast<double>(
            right_disparity_image[width_*y + x])/disparity_factor_ + 0.5);
      if (x + rightDisparityValue >= width_) {
        right_disparity_image[width_*y + x] = 0;
        continue;
      }

      int leftDisparityValue = static_cast<int>(static_cast<double>(
            left_disparity_image[width_*y + x+rightDisparityValue])/disparity_factor_ + 0.5);
      if (leftDisparityValue == 0 || abs(rightDisparityValue - leftDisparityValue) > consistency_threshold_) {
        right_disparity_image[width_*y + x] = 0;
      }
    }
  }
}

void SGMStereo::LoadRepresentationFromFile(const std::string& descriptors_path,
                                           DescriptorTensor* descriptors) {
  std::ifstream file(descriptors_path, std::ios::binary);

  int dims;
  file.read(reinterpret_cast<char*>(&dims), sizeof(dims));
  if (dims != 3) throw 1;
  std::vector<uint64_t> size;
  size.assign(dims, 0);
  for (int i = 0; i < dims; i++) {
    file.read(reinterpret_cast<char*>(&size[i]), sizeof(size[i]));
  }
  height_ = size[0];
  width_ = size[1];

  descriptors->resize(size[0]);
  for (uint64_t i = 0; i < size[0]; i++) {
    (*descriptors)[i].assign(size[1], Eigen::VectorXf(size[2]));
    for (uint64_t j = 0; j < size[1]; j++) {
      for (uint64_t k = 0; k < size[2]; k++)
        file.read(reinterpret_cast<char*>(&(*descriptors)[i][j][k]), sizeof((*descriptors)[i][j][k]));
    }
  }
}

} // namespace recon
