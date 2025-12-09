# YOLO Models Comprehensive Comparison Report

## Vehicle Detection Performance Analysis

**Date:** 2025-12-08 00:49:57  
**Models Compared:** YOLOv10n, YOLOv11n, YOLOv12n  
**Dataset:** Bangladesh Vehicle Detection (17,500 images)  
**Vehicle Classes:** 16 types  
**Test Set Size:** 1731 images  

---

## Executive Summary

### 🏆 Overall Winner: **YOLOv11**
- **Best Accuracy:** YOLOv11 (mAP@0.5:0.95 = 0.5621)
- **Fastest Model:** YOLOv11 (34.4 FPS)
- **Most Efficient:** YOLOv11 (mAP/MB = 0.1079)

---

## 1. Overall Performance Metrics

### Test Set Evaluation

|         |   mAP@0.5 |   mAP@0.5:0.95 |   mAP@0.75 |   Precision |   Recall |   Validation_Time_s |     FPS |       F1 |
|:--------|----------:|---------------:|-----------:|------------:|---------:|--------------------:|--------:|---------:|
| YOLOv10 |  0.770983 |       0.53653  |   0.606936 |    0.762611 | 0.703767 |             30.7201 | 56.3474 | 0.732009 |
| YOLOv11 |  0.803794 |       0.562052 |   0.642713 |    0.794603 | 0.737953 |             22.135  | 78.2019 | 0.765231 |
| YOLOv12 |  0.798722 |       0.557758 |   0.640359 |    0.788796 | 0.73537  |             22.6765 | 76.3345 | 0.761147 |

### Key Findings:
- **Highest mAP@0.5:0.95:** YOLOv11 achieved 0.5621
- **Speed Champion:** YOLOv11 runs at 34.4 FPS (batch=1)
- **Precision Leader:** YOLOv11 with 0.7946
- **Recall Leader:** YOLOv11 with 0.7380

---

## 2. Per-Class Performance Analysis

### Average Precision by Vehicle Class

|         |   bicycle |      bus |   bhotbhoti |      car |      cng |   easybike |   leguna |   motorbike |   pedestrian |   pickup |   powertiller |   rickshaw |   shoppingVan |    truck |      van |   wheelbarrow |
|:--------|----------:|---------:|------------:|---------:|---------:|-----------:|---------:|------------:|-------------:|---------:|--------------:|-----------:|--------------:|---------:|---------:|--------------:|
| YOLOv10 |  0.480356 | 0.65239  |    0.603141 | 0.648559 | 0.652765 |   0.645938 | 0.704835 |    0.425349 |     0.30663  | 0.443311 |      0.723982 |   0.625118 |      0.257864 | 0.51937  | 0.461717 |      0.433159 |
| YOLOv11 |  0.512843 | 0.669017 |    0.646103 | 0.665076 | 0.662091 |   0.648024 | 0.753292 |    0.432775 |     0.309511 | 0.475227 |      0.749803 |   0.633033 |      0.323262 | 0.537588 | 0.480938 |      0.494256 |
| YOLOv12 |  0.500171 | 0.662933 |    0.65277  | 0.661314 | 0.654904 |   0.643571 | 0.737494 |    0.425734 |     0.310154 | 0.47047  |      0.757273 |   0.625961 |      0.295361 | 0.560786 | 0.480506 |      0.484719 |

### Class-Specific Champions:

- **bicycle**: YOLOv11 (AP = 0.5128)
- **bus**: YOLOv11 (AP = 0.6690)
- **bhotbhoti**: YOLOv12 (AP = 0.6528)
- **car**: YOLOv11 (AP = 0.6651)
- **cng**: YOLOv11 (AP = 0.6621)
- **easybike**: YOLOv11 (AP = 0.6480)
- **leguna**: YOLOv11 (AP = 0.7533)
- **motorbike**: YOLOv11 (AP = 0.4328)
- **pedestrian**: YOLOv12 (AP = 0.3102)
- **pickup**: YOLOv11 (AP = 0.4752)
- **powertiller**: YOLOv12 (AP = 0.7573)
- **rickshaw**: YOLOv11 (AP = 0.6330)
- **shoppingVan**: YOLOv11 (AP = 0.3233)
- **truck**: YOLOv12 (AP = 0.5608)
- **van**: YOLOv11 (AP = 0.4809)
- **wheelbarrow**: YOLOv11 (AP = 0.4943)


### Bangladesh-Specific Vehicles Performance:

- **YOLOv10**: 0.6464
- **YOLOv11**: 0.6685
- **YOLOv12**: 0.6629


---

## 3. Speed & Efficiency Analysis

### Inference Speed Comparison

|         |   FPS_batch1 |   FPS_batch8 |   FPS_batch16 |
|:--------|-------------:|-------------:|--------------:|
| YOLOv10 |      33.9495 |      49.2168 |       50.5498 |
| YOLOv11 |      34.4048 |      48.0957 |       48.8048 |
| YOLOv12 |      28.3448 |      45.4377 |       47.1692 |

### Resource Utilization

|         |   GPU_Memory_MB |   Model_Size_MB |
|:--------|----------------:|----------------:|
| YOLOv10 |         75.9473 |         5.48425 |
| YOLOv11 |         74.625  |         5.21065 |
| YOLOv12 |         74.6021 |         5.25496 |

### Efficiency Scores

|         |   mAP_per_MB |   mAP_per_ms |
|:--------|-------------:|-------------:|
| YOLOv10 |    0.0978312 |    0.0182149 |
| YOLOv11 |    0.107866  |    0.0193373 |
| YOLOv12 |    0.106139  |    0.0158095 |

**Interpretation:**
- Higher `mAP_per_MB` = Better accuracy for model size
- Higher `mAP_per_ms` = Better accuracy for inference time

---

## 4. Statistical Significance Analysis

### 95% Confidence Intervals

| Model   |   Mean_mAP |   CI_Lower |   CI_Upper |   CI_Width |
|:--------|-----------:|-----------:|-----------:|-----------:|
| YOLOv10 |   0.537633 |   0.536678 |   0.538589 | 0.00191124 |
| YOLOv11 |   0.562213 |   0.561298 |   0.563127 | 0.00182917 |
| YOLOv12 |   0.557123 |   0.55621  |   0.558037 | 0.00182692 |

### Paired T-Test Results

| Comparison         |   t-statistic |      p-value | Significant   |
|:-------------------|--------------:|-------------:|:--------------|
| YOLOv10 vs YOLOv11 |     -37.6247  | 1.16866e-193 | Yes           |
| YOLOv10 vs YOLOv12 |     -29.2507  | 2.2963e-136  | Yes           |
| YOLOv11 vs YOLOv12 |       7.55879 | 9.18779e-14  | Yes           |

**ANOVA:** F-statistic = 752.4868, p-value = 0.0000

**Conclusion:** Statistically significant differences exist between models (p < 0.05)

---

## 5. NMS IoU Sensitivity Analysis

|          |   YOLOv10 |   YOLOv11 |   YOLOv12 |
|:---------|----------:|----------:|----------:|
| IoU@0.50 |   0.53653 |  0.561386 |  0.556439 |
| IoU@0.60 |   0.53653 |  0.561767 |  0.557733 |
| IoU@0.70 |   0.53653 |  0.562052 |  0.557758 |
| IoU@0.75 |   0.53653 |  0.56159  |  0.557811 |
| IoU@0.80 |   0.53653 |  0.5596   |  0.556173 |
| IoU@0.85 |   0.53653 |  0.553872 |  0.55119  |
| IoU@0.90 |   0.53653 |  0.536809 |  0.536198 |
| IoU@0.95 |   0.53653 |  0.460598 |  0.465695 |

#### Key Findings:
- **Stability:** YOLOv10 shows perfect stability (σ=0.000000) - completely flat performance
- **Sensitivity:** YOLOv11 is most affected by NMS threshold changes (highest variation)
- **Performance Drop:** YOLOv11 drops 17.85% from IoU@0.5 to IoU@0.95

#### Interpretation:
Flat response curves indicate **robustness** to NMS configuration changes - a desirable property for production deployment where NMS thresholds may need adjustment based on specific use cases.

---

## 6. Training Dynamics Summary


|         |   Total_Epochs |   Final_Train_Loss |   Final_Val_mAP50 |   Best_Val_mAP50 |   Best_Epoch |   Final_Precision |   Final_Recall |   Epochs_to_90pct |
|:--------|---------------:|-------------------:|------------------:|-----------------:|-------------:|------------------:|---------------:|------------------:|
| YOLOV10 |             50 |            2.22747 |           0.76946 |          0.76946 |           50 |           0.75978 |        0.70485 |                23 |
| YOLOV11 |             50 |            1.06146 |           0.7971  |          0.7971  |           50 |           0.77877 |        0.73299 |                20 |
| YOLOV12 |             50 |            1.06405 |           0.79749 |          0.7976  |           48 |           0.78509 |        0.73091 |                21 |

**Convergence Analysis:**
- Fastest convergence: YOLOV11
- Most epochs trained: YOLOV10

---

## 7. Deployment Recommendations

### For Real-Time Applications (Speed Priority):
**Recommended Model:** YOLOv11
- **Reason:** Highest FPS (34.4) enables real-time processing
- **Use Case:** Live traffic monitoring, dashcam applications
- **Trade-off:** Slight accuracy reduction (0.5621 mAP)

### For Accuracy-Critical Applications:
**Recommended Model:** YOLOv11
- **Reason:** Highest mAP@0.5:0.95 (0.5621)
- **Use Case:** Forensic analysis, detailed traffic studies
- **Trade-off:** Lower FPS (34.4)

### For Balanced Deployment:
**Recommended Model:** YOLOv11
- **Reason:** Best speed-accuracy trade-off (overall score: 1.0000)
- **Use Case:** General-purpose vehicle detection systems
- **Metrics:** mAP = 0.5621, FPS = 34.4

### For Edge Devices (Resource-Constrained):
**Recommended Model:** YOLOv11
- **Reason:** Highest efficiency score (mAP/MB = 0.1079)
- **Use Case:** Mobile devices, embedded systems
- **Model Size:** 5.2 MB

---

## 8. Key Visualizations

All visualizations are saved in: `e:\Projects\capstone-model-train-v2\output/ultimate_comparison\visualizations`

1. **Basic Metrics Comparison** - `01_basic_metrics_comparison.png`
2. **Per-Class Performance Heatmap** - `02_per_class_heatmap.png`
3. **Training Curves** - `03_training_curves.png`
4. **Speed Comparison** - `04_speed_comparison.png`
5. **IoU Robustness** - `05_iou_robustness.png`
6. **Confidence Intervals** - `06_confidence_intervals.png`
7. **Confusion Matrices** - `07_confusion_matrices.png`
8. **Speed-Accuracy Trade-off** - `08_speed_accuracy_tradeoff.png`
9. **Vehicle Category Performance** - `09_vehicle_category_performance.png`
10. **Comprehensive Dashboard** - `10_comprehensive_dashboard.png`

---

## 9. Research Questions Answered

### Q1: Which YOLO version achieves the highest mAP?
**Answer:** YOLOv11 with mAP@0.5:0.95 of 0.5621

### Q2: Are performance differences statistically significant?
**Answer:** Yes, p-value < 0.05 (ANOVA p = 0.0000)

### Q3: Which model offers the best speed-accuracy trade-off?
**Answer:** YOLOv11 (normalized score: 1.0000)

### Q4: Do models show different strengths across vehicle classes?
**Answer:** Yes, see per-class analysis. Different models excel at different vehicle types.

### Q5: Which model has superior localization accuracy?
**Answer:** YOLOv10 maintains best performance at high IoU thresholds (IoU@0.95)

### Q6: How do models perform on Bangladesh-specific vehicles?
**Answer:** Average AP for BD vehicles (CNG, rickshaw, etc.):
- YOLOv10: 0.6464
- YOLOv11: 0.6685
- YOLOv12: 0.6629


### Q7: What are the computational efficiency differences?
**Answer:** See efficiency scores - YOLOv11 offers best mAP per model size.

---

## 10. Limitations & Future Work

### Limitations:
1. Test set size: 1731 images (consider larger test sets for more robust conclusions)
2. Single hardware configuration (RTX 3070) - results may vary on different GPUs
3. Fixed input resolution (640x640) - multi-scale testing could provide additional insights

### Future Work Recommendations:
1. **Test-Time Augmentation (TTA):** Ensemble predictions with augmentations
2. **Post-Processing Optimization:** Tune NMS parameters per model
3. **Ensemble Methods:** Combine predictions from multiple models
4. **Domain-Specific Fine-Tuning:** Additional training on Bangladesh-specific scenarios
5. **Quantization:** Test INT8/FP16 versions for edge deployment
6. **Video Stream Testing:** Evaluate temporal consistency on video data

---

## 11. Conclusion

This comprehensive analysis compared three state-of-the-art YOLO versions (v10, v11, v12) across 16 vehicle classes using a rigorous evaluation methodology including:

- ✅ Standard detection metrics (mAP, Precision, Recall)
- ✅ Per-class performance analysis
- ✅ Speed benchmarking across multiple batch sizes
- ✅ Statistical significance testing
- ✅ IoU robustness evaluation
- ✅ Resource efficiency analysis
- ✅ Domain-specific performance (Bangladesh vehicles)

**Final Recommendation:**

Deploy **YOLOv11** for balanced performance, or choose specialized models based on your priority:
- **Accuracy:** YOLOv11
- **Speed:** YOLOv11
- **Efficiency:** YOLOv11

All models demonstrate strong performance on the Bangladesh vehicle dataset, with statistically significant differences.

---

## 12. Data Files

All metrics and results are saved as CSV files in `e:\Projects\capstone-model-train-v2\output/ultimate_comparison`:

1. `01_basic_metrics.csv` - Overall performance metrics
2. `02_per_class_ap.csv` - Per-class average precision
3. `03_speed_efficiency.csv` - Speed and efficiency metrics
4. `04_iou_robustness.csv` - Performance at different IoU thresholds
5. `05_size_performance.csv` - Size-based performance analysis
6. `06_confidence_intervals.csv` - Bootstrap confidence intervals
7. `07_statistical_tests.csv` - T-test and ANOVA results
8. `08_training_summary.csv` - Training dynamics summary

---

*Report generated by Ultimate YOLO Comparison Script*  
*Timestamp: 2025-12-08 00:49:57*  
*Total Analysis Time: [Completed]*
