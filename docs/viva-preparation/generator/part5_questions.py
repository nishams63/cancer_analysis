"""
PART 5 — VIVA QUESTIONS & ANSWERS
Contains 95 comprehensive viva questions divided into:
- Basic Questions (18)
- Intermediate Questions (18)
- 'Why' Questions (18)
- Practical & Scenario Questions (16)
- Rapid-Fire Questions (25)
All answers are grounded in our Personalized Precision Oncology project.
"""

VIVA_BASIC_QUESTIONS = [
    {
        "q": "What is the primary objective of your machine learning project?",
        "a": "The primary objective of our project is <strong>Personalized Precision Medicine for Oncology Treatment Optimization</strong>. We developed an autonomous multi-agent AI system that predicts adverse chemotherapy toxicity risk (Low, Moderate, High) in Stage 1 and forecasts tumor progression using digital pathology images and longitudinal blood biomarkers in Stage 2 to assist oncologists in making safer treatment decisions."
    },
    {
        "q": "What type of machine learning problem is your Stage 1 model solving?",
        "a": "It is a <strong>Supervised Multi-Class Classification</strong> problem. The model takes 25 clinical patient features and predicts one of three discrete risk categories: <code>Low</code>, <code>Moderate</code>, or <code>High</code> toxicity risk, along with class probabilities."
    },
    {
        "q": "What is the difference between Supervised and Unsupervised Learning?",
        "a": "<strong>Supervised Learning</strong> trains on labeled data where the ground-truth target is provided for each input (like our patient records with known toxicity outcomes). <strong>Unsupervised Learning</strong> trains on unlabeled data to discover natural groupings or compress features without predefined target labels (like clustering patient lab phenotypes using K-Means)."
    },
    {
        "q": "What is a Feature and what is a Target in Machine Learning?",
        "a": "A <strong>Feature (X)</strong> is an independent input variable describing the observation (e.g., patient age, creatinine level, drug dose). The <strong>Target (y)</strong> is the dependent ground-truth variable the model is trained to predict (e.g., toxicity risk category)."
    },
    {
        "q": "Why do we split data into Training, Validation, and Testing sets?",
        "a": "We split data to guarantee that our model learns generalizable patterns rather than memorizing data. The <strong>Training set</strong> fits model weights; the <strong>Validation set</strong> tunes hyperparameters and checks for overfitting; and the <strong>Testing set</strong> is locked away to provide an unbiased final benchmark of real-world generalization."
    },
    {
        "q": "What is Overfitting and how do you detect it?",
        "a": "<strong>Overfitting</strong> occurs when a model learns the training data and noise too closely, failing to generalize to unseen test data. It is detected when the model achieves very high training accuracy (e.g., 99%) but significantly lower validation/testing accuracy (e.g., 55%)."
    },
    {
        "q": "What is Underfitting and how can it be resolved?",
        "a": "<strong>Underfitting</strong> occurs when a model is too simple to capture underlying patterns, resulting in poor accuracy on both training and test data. It is resolved by adding more informative features, reducing regularization, or switching to a higher-capacity algorithm (e.g., moving from linear models to gradient boosted trees)."
    },
    {
        "q": "What is an Artificial Neuron?",
        "a": "An artificial neuron is the core mathematical unit of a neural network. It computes the weighted sum of its inputs (Σ wᵢxᵢ), adds a bias offset (b), and passes the result through a non-linear activation function (like ReLU or Sigmoid) to generate an output signal."
    },
    {
        "q": "What is the purpose of an Activation Function?",
        "a": "The purpose of an activation function is to introduce <strong>non-linearity</strong> into the network. Without non-linear activations, stacking multiple neural network layers mathematically collapses into a single linear regression model, incapable of learning complex, non-linear patterns."
    },
    {
        "q": "What is ReLU and why is it so widely used?",
        "a": "<strong>ReLU (Rectified Linear Unit)</strong> is defined as <code>f(x) = max(0, x)</code>. It outputs zero for negative inputs and remains linear for positive inputs. It is widely used because it computes extremely fast and avoids the vanishing gradient problem for positive values, accelerating neural network convergence."
    },
    {
        "q": "What is Epoch, Batch, and Iteration?",
        "a": "An <strong>Epoch</strong> is one complete pass of the entire training dataset through the network. A <strong>Batch</strong> is the subset of training samples processed concurrently before updating weights. An <strong>Iteration</strong> is a single weight update step (Total Iterations per Epoch = Total Samples / Batch Size)."
    },
    {
        "q": "What is Gradient Descent in simple terms?",
        "a": "Gradient descent is an optimization algorithm that iteratively adjusts neural network weights in the direction that reduces prediction error (stepping downhill along the negative slope of the loss function) to reach the minimum cost."
    },
    {
        "q": "What is the Learning Rate?",
        "a": "The learning rate is a hyperparameter that controls the step size the optimizer takes when updating weights during gradient descent. If too large, training diverges; if too small, training converges impractically slowly."
    },
    {
        "q": "What is a Confusion Matrix?",
        "a": "A confusion matrix is a table that summarizes the performance of a classification model by displaying the counts of True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN) across all target classes."
    },
    {
        "q": "What is the difference between Precision and Recall?",
        "a": "<strong>Precision</strong> answers: 'Out of all cases predicted positive, how many were truly positive?' (TP / (TP + FP)). <strong>Recall (Sensitivity)</strong> answers: 'Out of all actual positive cases, how many did the model correctly identify?' (TP / (TP + FN))."
    },
    {
        "q": "What is an ML Pipeline?",
        "a": "An ML pipeline is an automated sequence that bundles data ingestion, preprocessing (imputation, scaling, encoding), and model estimation into an atomic, reproducible workflow, guaranteeing consistent transformations and preventing data leakage."
    },
    {
        "q": "What is Data Preprocessing?",
        "a": "Data preprocessing is the process of cleaning, transforming, and formatting raw, messy real-world data (handling missing entries, encoding text categories, scaling numbers) into numerical feature matrices ready for machine learning algorithms."
    },
    {
        "q": "What is the difference between Model Parameters and Hyperparameters?",
        "a": "<strong>Parameters</strong> are internal variables learned automatically from data during training (weights, biases, tree split values). <strong>Hyperparameters</strong> are external configuration settings chosen manually by the engineer before training begins (learning rate, tree depth, batch size)."
    }
]

VIVA_INTERMEDIATE_QUESTIONS = [
    {
        "q": "Which machine learning algorithm was chosen as your champion model for Stage 1, and why?",
        "a": "We selected <strong>Candidate V4: Regularized LightGBM (Gradient Boosted Decision Trees)</strong>. It outperformed Logistic Regression and Random Forest on our tabular dataset (1,750 encounters) by naturally handling mixed categorical and numerical clinical features, offering fast training, robust handling of skewed lab values, and allowing conservative decision threshold tuning to prioritize High-Risk Recall."
    },
    {
        "q": "What performance did Candidate V4 achieve on your locked test set?",
        "a": "Candidate V4 achieved an overall <strong>Accuracy of 0.5766</strong>, a <strong>Macro F1-score of 0.5288</strong> (with a 95% bootstrap confidence interval of [0.5035, 0.5521]), and a critical <strong>High-Risk Safety Recall of 0.6287</strong> (95% CI: [0.5759, 0.6783]), confirming balanced performance without overfitting."
    },
    {
        "q": "Why is Accuracy an insufficient metric for evaluating your oncology toxicity model?",
        "a": "Accuracy is misleading because clinical datasets frequently exhibit class imbalance (e.g., only 22% of encounters are High Risk). A naive model predicting 'Low Risk' for all patients could achieve 78% accuracy while failing to identify a single high-risk patient, which would be fatal in oncology. We use <strong>Macro F1</strong> and <strong>High-Risk Recall</strong> to ensure all risk tiers are safely identified."
    },
    {
        "q": "How did you prevent data leakage during data splitting and preprocessing?",
        "a": "We enforced a strict <strong>Patient-Level Split</strong> based on <code>patient_id</code> (700 Train, 150 Val, 150 Test). This ensured that all historical visits for any individual patient existed solely in one split. Furthermore, all imputers and scalers were <strong>fitted strictly on training data</strong> and merely applied to test data using <code>transform()</code>."
    },
    {
        "q": "How does Random Forest work, and how does it prevent overfitting compared to a single Decision Tree?",
        "a": "Random Forest uses <strong>Bagging (Bootstrap Aggregation)</strong> and feature subsampling. It trains hundreds of individual decision trees on different random subsets of data and features. Each tree overfits slightly in different ways, but when their predictions are averaged (majority voting), individual variance cancels out, drastically reducing overall overfitting."
    },
    {
        "q": "Explain the Bias-Variance tradeoff in the context of your project.",
        "a": "A simple linear model had <strong>High Bias</strong> (underfitting, failing to capture complex non-linear lab interactions, yielding 46% F1). An unpruned decision tree had <strong>High Variance</strong> (memorizing individual patient records, yielding 99% train accuracy but 48% test accuracy). LightGBM with tree depth 5 and L2 regularization achieved the optimal tradeoff."
    },
    {
        "q": "What is K-Fold Cross-Validation, and which variant did you use?",
        "a": "K-Fold CV splits training data into K subsets, training on K-1 folds and validating on the remaining fold, repeating K times. We used <strong>5-Fold Stratified Group K-Fold</strong>: 'Stratified' preserved the Low/Mod/High class proportions in every fold, and 'Group' ensured all visits of a patient remained in the same fold."
    },
    {
        "q": "How does Logistic Regression perform multi-class classification?",
        "a": "Logistic Regression handles multi-class classification using either <strong>One-vs-Rest (OvR)</strong>, where a binary model is trained for each class against all others, or <strong>Multinomial Logistic Regression</strong>, which uses the Softmax function to output a normalized probability distribution across all classes simultaneously."
    },
    {
        "q": "What is the difference between L1 and L2 Regularization?",
        "a": "<strong>L1 Regularization (Lasso)</strong> adds the sum of absolute weights (|W|) to the loss, driving irrelevant feature weights to absolute zero (acting as automated feature selection). <strong>L2 Regularization (Ridge)</strong> adds the sum of squared weights (W²), penalizing large weights and shrinking them smoothly toward zero without eliminating them."
    },
    {
        "q": "How does Backpropagation calculate weight updates in a neural network?",
        "a": "Backpropagation computes the gradient of the loss function with respect to every weight by systematically applying the <strong>calculus Chain Rule</strong> backward from the output layer to the input layer (∂Loss/∂Weight = ∂Loss/∂Output * ∂Output/∂Net * ∂Net/∂Weight). These gradients are then passed to the optimizer."
    },
    {
        "q": "What is the Vanishing Gradient Problem in deep neural networks?",
        "a": "The vanishing gradient problem occurs when error gradients shrink exponentially as they propagate backward through many layers (especially when using Sigmoid or Tanh activations whose derivatives are < 0.25). As a result, early layer weights receive virtually zero updates and stop learning. It is solved using <strong>ReLU activations</strong>, Residual connections (ResNets), and Batch Normalization."
    },
    {
        "q": "Why are CNNs preferred over standard feedforward neural networks (MLPs) for medical images?",
        "a": "If a 224x224 RGB image is flattened into an MLP, it creates 150,528 inputs per neuron, destroying 2D spatial relationships and resulting in millions of parameters that quickly overfit. CNNs preserve 2D spatial context using <strong>local receptive fields</strong>, <strong>parameter sharing</strong> (reusing the same filter across the image), and <strong>pooling</strong>, making them vastly more efficient and translation invariant."
    },
    {
        "q": "How does an LSTM cell capture long-term dependencies compared to a standard RNN?",
        "a": "A standard RNN only has a recurrent hidden state that suffers from vanishing gradients over long sequences. An LSTM introduces an uninterrupted <strong>Cell State pipeline</strong> regulated by three gates: <strong>Forget Gate</strong> (removes obsolete memory), <strong>Input Gate</strong> (adds new relevant input), and <strong>Output Gate</strong> (emits current state), allowing gradients to flow backward through time without vanishing."
    },
    {
        "q": "What is the role of the Adam Optimizer?",
        "a": "Adam (Adaptive Moment Estimation) combines the advantages of Momentum (tracking the exponentially decaying moving average of past gradients to maintain velocity) and RMSProp (tracking the moving average of squared gradients to adapt individual learning rates for each parameter), providing fast, stable convergence."
    },
    {
        "q": "What is Dropout and why is it disabled during inference?",
        "a": "Dropout is a regularization technique that randomly zeroes out a fraction of neurons (e.g., 40%) during training iterations to prevent feature co-adaptation. It is disabled during inference because at test time we want the full deterministic network ensemble to predict using all learned features, scaled by the retention probability."
    },
    {
        "q": "What is Data Imbalance and how can it be handled?",
        "a": "Data imbalance occurs when certain target classes have significantly fewer samples than others. It can be addressed via <strong>resampling techniques</strong> (SMOTE oversampling of minority class, random undersampling), <strong>algorithmic adjustments</strong> (class-weighted loss functions like class_weight='balanced'), or tuning decision thresholds."
    },
    {
        "q": "What is Softmax and why is it used in multi-class classification?",
        "a": "Softmax transforms a vector of K raw numerical logits into a valid probability distribution where each value is between 0 and 1, and the sum of all K values equals exactly 1.0 (<code>P(y=k) = e^z_k / Σ e^z_j</code>), making it directly interpretable as class probabilities."
    },
    {
        "q": "What is the difference between Pearson and Spearman correlation?",
        "a": "<strong>Pearson correlation</strong> measures the strength of a strictly <em>linear</em> relationship between two continuous variables and is sensitive to outliers. <strong>Spearman correlation</strong> evaluates <em>monotonic</em> relationships based on ranked values, making it non-parametric and robust to non-linear associations and outliers."
    }
]

VIVA_WHY_QUESTIONS = [
    {
        "q": "Why do we scale features before feeding them into algorithms like KNN, SVM, or Neural Networks?",
        "a": "Because distance- and gradient-based algorithms calculate Euclidean distances or parameter gradients. If one feature (like platelet count) has values in hundreds of thousands while another (like creatinine) has values near 1.0, the large-magnitude feature will completely dominate the mathematical distance, causing the model to ignore the smaller biomarker regardless of its clinical importance."
    },
    {
        "q": "Why do tree-based models (like Decision Trees, Random Forest, LightGBM) NOT require feature scaling?",
        "a": "Tree-based algorithms make split decisions based on the relative ordering of values within an individual feature (e.g., 'Is creatinine > 1.4 mg/dL?'). Because each split evaluates one feature at a time independently, scaling or shifting a feature does not alter the sorting order or the information gain calculation."
    },
    {
        "q": "Why did you use Median Imputation instead of Mean Imputation for patient laboratory biomarkers?",
        "a": "Clinical lab values (such as creatinine, liver enzymes, and tumor markers) are frequently right-skewed and contain extreme physiological spike outliers. The <strong>mean is heavily pulled by outliers</strong>, which would artificially distort imputed values. The <strong>median represents the true 50th percentile</strong> and is completely robust against outlier distortion."
    },
    {
        "q": "Why is patient-level data splitting strictly required in healthcare ML instead of random row splitting?",
        "a": "Cancer patients often have multiple hospital visits and encounters in the dataset. If we perform a random row split, visit #1 of a patient could be in the training set and visit #2 in the test set. The model would memorize patient-specific baseline genetics and lab quirks, creating severe <strong>data leakage</strong> and giving an artificially inflated test score that fails completely on new patients."
    },
    {
        "q": "Why does overfitting happen in machine learning and deep learning models?",
        "a": "Overfitting happens when a model has excessive capacity (too many parameters, trees that are too deep, or neural networks with too many layers) relative to the amount of training data. The model finds mathematical shortcuts and memorizes noise, specific patient IDs, or artifacts in the training set rather than genuine generalizable biological patterns."
    },
    {
        "q": "Why is High-Risk Recall prioritized over overall Accuracy in your oncology project?",
        "a": "In oncology, a <strong>False Negative</strong> (predicting Low Risk when a patient is actually High Risk) is catastrophic: the patient receives a full chemotherapy dose without protective monitoring, leading to life-threatening toxicity or organ failure. A False Positive merely results in extra monitoring. Therefore, clinical safety requires maximizing High-Risk Recall (0.6287)."
    },
    {
        "q": "Why do we use the F1-score rather than the arithmetic mean of Precision and Recall?",
        "a": "The arithmetic mean treats Precision and Recall independently: if Precision is 1.0 and Recall is 0.0, the arithmetic mean is 0.5 (which hides complete failure). The <strong>F1-score is the Harmonic Mean</strong> (2 * P * R / (P + R)). The harmonic mean penalizes extreme imbalances heavily and approaches zero if either metric is close to zero."
    },
    {
        "q": "Why do we use Cross-Entropy Loss instead of Mean Squared Error for classification?",
        "a": "When combined with Sigmoid or Softmax activations, Mean Squared Error produces a non-convex loss landscape with many flat regions (where derivatives approach zero), causing gradient descent to stall. <strong>Cross-Entropy Loss</strong> is convex for logistic models and penalizes confident wrong predictions with an exponentially increasing, logarithmic penalty (-log(p)), generating strong corrective gradients."
    },
    {
        "q": "Why do we use One-Hot Encoding for cancer types but Ordinal Encoding for cancer stages?",
        "a": "<code>cancer_type</code> (NSCLC, Colorectal, Breast) is a <strong>nominal variable</strong> with no inherent mathematical ranking; assigning 1, 2, 3 would falsely trick the algorithm into treating Breast Cancer as 'greater than' Lung Cancer. <code>cancer_stage</code> (Stage I, II, III, IV) is an <strong>ordinal variable</strong> with a true progressive biological severity hierarchy, so ordinal integers (1 to 4) preserve meaningful order."
    },
    {
        "q": "Why do we use Max Pooling in Convolutional Neural Networks?",
        "a": "Max pooling downsamples feature maps by extracting the maximum activation in each window (e.g., 2x2). This achieves three critical benefits: 1. It halves the spatial dimensions, drastically reducing parameter count and GPU computation; 2. It provides <strong>translation invariance</strong> (a tumor cell is detected even if shifted slightly); 3. It expands the receptive field of subsequent convolutional layers."
    },
    {
        "q": "Why do we need Backpropagation in neural network training?",
        "a": "Without backpropagation, calculating the derivative of the loss function with respect to millions of individual weights would require numerical finite-difference approximations (taking hours or days per step). Backpropagation uses the calculus chain rule to compute exact analytical gradients for all parameters simultaneously in a single backward pass through the network."
    },
    {
        "q": "Why is Adam considered superior to basic Stochastic Gradient Descent for complex networks?",
        "a": "Basic SGD uses a single, fixed global learning rate for all parameters, causing slow progress along flat plateaus and wild oscillations across steep ravines. <strong>Adam automatically computes per-parameter adaptive learning rates</strong> by maintaining rolling averages of past gradients (momentum) and past squared gradients (scale adaptation), navigating complex non-convex surfaces efficiently."
    },
    {
        "q": "Why do we freeze model weights before evaluating on the locked test set?",
        "a": "To eliminate <strong>test set contamination and p-hacking</strong>. If an engineer repeatedly evaluates and modifies hyperparameters on the test set, information from the test set leaks into model design decisions. Freezing the model artifact ensures the locked test set serves as a true, unbiased audit of real-world generalization."
    },
    {
        "q": "Why is late fusion used in multimodal deep learning systems?",
        "a": "In late fusion, individual modalities (e.g., 2D histology images and 1D temporal biomarker sequences) are first processed by specialized domain-specific neural backbones (CNN and LSTM). Their high-level score representations or embeddings are fused at the final stage. This allows each model to specialize in its native data format without forcing incompatible raw pixel and lab signals to merge early."
    },
    {
        "q": "Why do we calculate 95% Bootstrap Confidence Intervals for evaluation metrics?",
        "a": "A single point-estimate metric (e.g., Accuracy = 57.6%) does not reveal whether the score was a fluke of the specific test sample. <strong>Bootstrap resampling</strong> (resampling the test set with replacement 1,000 times) estimates the sampling distribution, proving statistical confidence (e.g., [0.5520, 0.6000]) and assuring clinicians that performance is stable."
    },
    {
        "q": "Why did we implement an independent Evaluation Engineer role?",
        "a": "To ensure objective safety verification. Developers naturally suffer from confirmation bias and focus on demonstrating model success. An independent Evaluation Engineer rigorously audits the model against edge cases, vulnerable demographic cohorts, and clinical failure modes before deployment."
    },
    {
        "q": "Why is an API schema validation framework like Pydantic essential for clinical model deployment?",
        "a": "If a hospital client sends a string instead of a float for a drug dose, or sends a negative creatinine value, an unvalidated ML model could crash the server or output a dangerously wrong risk score. Pydantic validates data types, physiological boundaries, and required fields at the API doorway before execution."
    },
    {
        "q": "Why is R² (R-squared) used to evaluate regression models?",
        "a": "Because MAE and RMSE depend on the scale of the target variable (e.g., 0.2 ng/mL vs 500 mg). <strong>R² is a scale-independent metric</strong> that quantifies the proportion of target variance explained by the model compared to a naive horizontal baseline (mean target value). R² = 1.0 means perfect prediction."
    }
]

VIVA_PRACTICAL_QUESTIONS = [
    {
        "q": "Which Python libraries did you use in your project and for what purpose?",
        "a": "We used: <strong>Pandas & NumPy</strong> for data wrangling and numerical array operations; <strong>Pydantic</strong> for schema validation; <strong>Matplotlib & Seaborn</strong> for EDA distribution plots and correlation heatmaps; <strong>Scikit-Learn</strong> for preprocessing pipelines, metrics, and baseline models; <strong>LightGBM</strong> for Candidate V4 gradient boosted trees; <strong>PyTorch</strong> for Stage 2 CNN and LSTM deep networks; <strong>FastAPI & Uvicorn</strong> for REST API microservice integration; and <strong>PyTest</strong> for automated test suites."
    },
    {
        "q": "How would you handle missing values in a real-time production inference API?",
        "a": "In real-time inference, the incoming JSON payload cannot use test-set statistics. We load our <strong>pre-fitted preprocessor pipeline</strong> (serialized as <code>preprocessor.joblib</code>), which automatically replaces missing numerical fields with the precomputed medians from the training cohort, and assigns unknown categories to <code>'Unknown'</code>."
    },
    {
        "q": "How would you detect and handle outliers in clinical laboratory data?",
        "a": "We detect outliers using box plots and the <strong>Interquartile Range (IQR)</strong> method (values outside Q1 - 1.5*IQR to Q3 + 1.5*IQR) and Z-scores (|Z| > 3). In our pipeline, we do not arbitrarily delete clinical outliers because an extreme biomarker often represents acute organ failure. Instead, we use robust algorithms (LightGBM) and robust median imputation that are resilient to extreme values."
    },
    {
        "q": "How do you select which ML algorithm to use for a new problem?",
        "a": "We follow a 4-step selection framework: 1. <strong>Analyze data modality:</strong> Tabular structured data &rarr; Gradient Boosted Trees (LightGBM/XGBoost); Images/Video &rarr; CNNs; Sequential text/time series &rarr; LSTMs/Transformers. 2. <strong>Analyze dataset size:</strong> Small datasets &rarr; Linear/Logistic models or Trees; Large datasets &rarr; Deep learning. 3. <strong>Interpretability requirement:</strong> Clinical applications need feature importances. 4. <strong>Benchmark multiple candidates:</strong> Compare baseline models against an independent validation set."
    },
    {
        "q": "How would you evaluate a classification model on an imbalanced dataset?",
        "a": "I would avoid relying on overall Accuracy. Instead, I would inspect the full <strong>Confusion Matrix</strong>, report class-specific <strong>Precision and Recall</strong>, compute <strong>Macro F1-score</strong> (which weights all classes equally), and plot the <strong>Precision-Recall Curve (PR-AUC)</strong> and ROC-AUC."
    },
    {
        "q": "What happens if a model performs very well on training data (98% accuracy) but poorly on testing data (52% accuracy)?",
        "a": "This is a textbook case of <strong>Overfitting (High Variance)</strong>. To fix it: 1. Reduce model complexity (restrict tree depth, prune branches); 2. Apply regularization (L2 weight decay, dropout); 3. Apply feature selection to eliminate noisy variables; 4. Perform cross-validation; 5. Collect or augment more training data."
    },
    {
        "q": "How do you serialize and save a trained model for deployment in Python?",
        "a": "In Scikit-Learn/LightGBM, we use <code>joblib.dump(model, 'model.joblib')</code> to serialize the trained model and preprocessing pipeline to disk. In PyTorch, we save model weights using <code>torch.save(model.state_dict(), 'model.pt')</code>. At serving time, the API loads the artifact once on startup using <code>joblib.load()</code> or <code>model.load_state_dict()</code>."
    },
    {
        "q": "How does your FastAPI service handle batch predictions from hospital wards?",
        "a": "We exposed a <code>POST /predict/batch</code> endpoint that accepts an array of patient encounter records. The service validates each record with Pydantic, converts the list into a single Pandas DataFrame, runs vectorized inference through the pipeline in one batch, and returns a JSON list of risk predictions and probabilities in under 30 milliseconds."
    },
    {
        "q": "What is Subgroup Analysis and why did your project evaluate 26 clinical cohorts?",
        "a": "Subgroup analysis evaluates whether the model performs equitably across different demographic and clinical subsets (e.g., elderly vs young patients, different cancer types, different stages). An overall accuracy of 58% might hide a failure rate of 85% in Stage IV lung cancer patients. Benchmarking across 26 cohorts certified that Candidate V4 maintained safe recall across all patient sub-populations."
    },
    {
        "q": "What is the difference between Data Drift and Concept Drift in production monitoring?",
        "a": "<strong>Data Drift (Covariate Shift)</strong> occurs when the distribution of input features changes over time (e.g., a hospital adopts a new laboratory assay that produces higher baseline creatinine readings). <strong>Concept Drift</strong> occurs when the statistical relationship between features and the target changes (e.g., a new chemotherapy protocol is introduced, altering which biomarkers lead to toxicity)."
    },
    {
        "q": "What is Data Augmentation in Deep Learning and why did you apply it to pathology tiles?",
        "a": "Data augmentation applies random, label-preserving transformations (horizontal flips, vertical flips, 90° rotations, subtle brightness shifts) to training images. We applied it to our 224x224 biopsy tiles to simulate different microscope orientations and staining variances, preventing the CNN from memorizing fixed pixel alignments and increasing generalization."
    },
    {
        "q": "What is Early Stopping and how does it work?",
        "a": "Early stopping is a regularization technique that monitors validation loss at the end of each training epoch. If validation loss does not improve for a predefined number of consecutive epochs (called <code>patience</code>, e.g., 7 epochs), training terminates automatically and the best checkpoint weights are restored."
    },
    {
        "q": "How do you convert model probability outputs into a final clinical decision?",
        "a": "By default, multi-class models predict the class with the highest probability (argmax). In clinical risk prediction, we tune <strong>decision thresholds</strong>: for example, if the predicted probability of High Toxicity exceeds 0.35 (rather than standard 0.50), the system conservatively flags the patient for high-risk monitoring to prioritize patient safety."
    },
    {
        "q": "What is the difference between ETL and ELT?",
        "a": "In <strong>ETL</strong>, data is transformed on an intermediate processing server before being loaded into the destination database. In <strong>ELT</strong>, raw data is loaded directly into modern, scalable cloud data warehouses (Snowflake, BigQuery) and transformed using in-database SQL engines."
    },
    {
        "q": "What automated tests did you write to verify your pipeline?",
        "a": "We wrote 40 automated unit and integration tests using <strong>PyTest</strong>. These tests verify: 1. Data schema validity and zero patient leakage; 2. Preprocessor output dimensions; 3. Model serialization/deserialization; 4. API endpoint response formats and HTTP status codes (200 OK, 422 Unprocessable Entity); and 5. Multimodal late fusion score boundaries."
    },
    {
        "q": "What is the clinical disclaimer for your AI system?",
        "a": "Our system is a <strong>machine learning research decision-support prototype</strong>. It is designed to assist oncologists by highlighting potential toxicity risks and progression indicators, but it is not approved for autonomous diagnosis or treatment prescription. All model outputs must be validated by licensed medical specialists."
    }
]

VIVA_RAPID_FIRE_QUESTIONS = [
    {
        "q": "1. What is Machine Learning?",
        "a": "Algorithms learning statistical patterns from historical data to make predictions without explicit hardcoded rules."
    },
    {
        "q": "2. What is Deep Learning?",
        "a": "A subset of ML based on multi-layered neural networks that automatically learns feature representations from raw data."
    },
    {
        "q": "3. What was the target variable in Stage 1 of your project?",
        "a": "<code>toxicity_risk</code>, with three classes: Low, Moderate, and High."
    },
    {
        "q": "4. What was your final champion ML model?",
        "a": "Candidate V4: Regularized LightGBM (Gradient Boosted Decision Trees)."
    },
    {
        "q": "5. What was Candidate V4's primary safety metric score?",
        "a": "High-Risk Safety Recall of <strong>0.6287</strong> on the locked test set."
    },
    {
        "q": "6. What is the formula for Accuracy?",
        "a": "Accuracy = (True Positives + True Negatives) / Total Samples."
    },
    {
        "q": "7. What is Recall?",
        "a": "Recall = TP / (TP + FN); the fraction of actual positive cases successfully caught."
    },
    {
        "q": "8. What is Precision?",
        "a": "Precision = TP / (TP + FP); the fraction of predicted positive cases that are truly positive."
    },
    {
        "q": "9. What is F1-Score?",
        "a": "The harmonic mean of Precision and Recall: 2 * (Precision * Recall) / (Precision + Recall)."
    },
    {
        "q": "10. Why is patient-level splitting crucial?",
        "a": "To ensure 0% data leakage across repeat visits of the same patient between train and test sets."
    },
    {
        "q": "11. What is the difference between Min-Max Normalization and Standardization?",
        "a": "Normalization bounds data to [0, 1]; Standardization centers data to mean 0 and standard deviation 1."
    },
    {
        "q": "12. What activation function is standard for hidden layers in modern neural networks?",
        "a": "ReLU (Rectified Linear Unit), f(x) = max(0, x)."
    },
    {
        "q": "13. What activation function is used on the output layer for multi-class classification?",
        "a": "Softmax, which outputs probabilities across all classes summing to 1.0."
    },
    {
        "q": "14. What does Backpropagation do?",
        "a": "Computes partial derivatives of the loss function with respect to weights using the calculus chain rule."
    },
    {
        "q": "15. What does the Optimizer do?",
        "a": "Updates network weights in the direction that minimizes loss (e.g., Adam, SGD)."
    },
    {
        "q": "16. What is Dropout?",
        "a": "A regularization technique that randomly shuts down a fraction of neurons during training to prevent co-adaptation."
    },
    {
        "q": "17. What are the two core operations in a CNN?",
        "a": "Convolution (filtering spatial features) and Pooling (downsampling dimensions)."
    },
    {
        "q": "18. What are the three gates inside an LSTM cell?",
        "a": "Forget Gate, Input Gate, and Output Gate."
    },
    {
        "q": "19. What does the Data Engineer deliver in your project?",
        "a": "Cleaned, schema-validated master patient datasets with zero-leakage partitions."
    },
    {
        "q": "20. What does the EDA Engineer do before model training?",
        "a": "Analyzes distributions, detects anomalies/outliers, and computes feature-target correlations."
    },
    {
        "q": "21. What is the difference between a Data Scientist and an ML Engineer?",
        "a": "Data Scientists focus on statistical exploration and prototype models; ML Engineers focus on production pipelines, latency, and APIs."
    },
    {
        "q": "22. What does an Evaluation Engineer do?",
        "a": "Independently benchmarks locked models, computes bootstrap confidence intervals, and audits subgroup fairness."
    },
    {
        "q": "23. What framework did the Integration Engineer use to deploy your model?",
        "a": "FastAPI, creating REST endpoints with Pydantic schema validation."
    },
    {
        "q": "24. What is Late Fusion in multimodal AI?",
        "a": "Processing individual modalities (images, time series) through separate neural backbones and combining their final output scores."
    },
    {
        "q": "25. What is the golden rule of clinical AI safety?",
        "a": "A False Negative in high-risk disease or toxicity is far more dangerous than a False Positive."
    }
]
