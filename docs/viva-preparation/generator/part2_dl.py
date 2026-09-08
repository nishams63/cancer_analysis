"""
PART 2 — DEEP LEARNING
Contains all 27 concepts formatted strictly with:
Simple definition, How it works, Small example (Stage 2 Oncology DL project), Viva-ready answer, Key point to remember.
"""

DL_CONCEPTS = [
    {
        "num": 1,
        "title": "What is Deep Learning?",
        "def": "Deep Learning (DL) is a specialized subset of Machine Learning based on Artificial Neural Networks with multiple stacked hidden layers ('deep' architectures). Unlike classical ML where human engineers must manually extract features, Deep Learning models automatically learn hierarchical representations directly from raw data like images, audio, and sequential clinical time series.",
        "how": "Data passes through successive layers of mathematical neurons. Early layers detect basic low-level features (e.g., edges, pixel gradients), intermediate layers assemble these into textures and patterns, and deep layers recognize high-level semantic concepts (e.g., malignant tumor cell nuclei).",
        "example": "In Stage 2 of our project, instead of manually measuring cell diameters on a microscope slide, our Deep Learning CNN directly ingests raw 224x224 RGB biopsy image tiles and learns to classify tissue as benign, malignant, or inflammatory.",
        "viva": "Deep Learning is a subfield of ML that uses multi-layered artificial neural networks to automatically perform feature extraction and pattern recognition directly from high-dimensional raw data such as digital pathology images and longitudinal biomarker sequences.",
        "remember": "Deep Learning = Artificial Neural Networks with multiple hidden layers + Automatic End-to-End Feature Representation Learning."
    },
    {
        "num": 2,
        "title": "ML vs DL (Representation Learning)",
        "def": "Classical Machine Learning relies heavily on manual feature engineering by human experts, and its performance typically plateaus as data volume increases. Deep Learning performs automatic feature learning directly from raw inputs and its performance scales dramatically with massive datasets and computational power (GPUs/TPUs).",
        "how": "In classical ML: Raw Data &rarr; Human Feature Engineering &rarr; ML Algorithm &rarr; Prediction. In Deep Learning: Raw Data &rarr; Deep Neural Network (Feature Learning + Classification in one end-to-end model) &rarr; Prediction.",
        "example": "In Stage 1 (ML), we manually calculated ratios like <code>creatinine / drug_dose</code> for tabular clinical data. In Stage 2 (DL), our CNN takes 224x224x3 raw pixel matrices (150,528 numbers per image) and extracts features autonomously through convolutional filters.",
        "viva": "Classical ML requires manual feature engineering and works well on small-to-medium structured tabular datasets; Deep Learning automatically extracts hierarchical representations from raw unstructured data (images, text, sequences) but requires large datasets and GPU hardware.",
        "remember": "Classical ML = Handcrafted features on tabular data. Deep Learning = Automatic representation learning on raw unstructured data."
    },
    {
        "num": 3,
        "title": "Neural Networks",
        "def": "An Artificial Neural Network (ANN) is a computational model inspired by the biological neural networks of the human brain. It consists of interconnected processing units called artificial neurons arranged in layers, which transmit signals to one another through weighted connections.",
        "how": "Each neuron receives inputs from preceding neurons, multiplies them by connection weights, adds a bias term, passes the sum through a non-linear activation function, and broadcasts the output signal to the subsequent layer.",
        "example": "In our project, we use an ANN to model non-linear interactions between complex longitudinal patient vitals and drug concentrations that simple linear equations fail to capture.",
        "viva": "A neural network is an interconnected network of computational nodes organized in layers that map non-linear relationships between complex inputs and target outputs by learning optimal connection weights.",
        "remember": "Neural networks are universal function approximators: given enough hidden neurons and non-linear activations, they can approximate any continuous mathematical function."
    },
    {
        "num": 4,
        "title": "Artificial Neuron (Perceptron)",
        "def": "The Artificial Neuron (Perceptron) is the fundamental building block of a neural network. It takes multiple numerical inputs, calculates their weighted sum, adds a bias, and passes the result through an activation function to generate an output.",
        "how": "Mathematically: z = (w₁x₁ + w₂x₂ + ... + wₙxₙ) + b = Σ(wᵢxᵢ) + b. Output: a = σ(z), where σ is a non-linear activation function (like ReLU or Sigmoid).",
        "example": "A single neuron in our toxicity network might take inputs x₁ = creatinine, x₂ = drug_dose, multiply them by weights w₁ = 0.85, w₂ = 0.65, add bias b = -1.2, and pass through Sigmoid to output a probability of acute renal stress.",
        "viva": "An artificial neuron calculates the dot product of input vectors and weight vectors, adds a bias offset, and applies a non-linear activation function to decide the neuron's activation strength.",
        "remember": "Equation to write on board: <strong>y = σ(WᵀX + b)</strong>."
    },
    {
        "num": 5,
        "title": "Input Layer, Hidden Layer, and Output Layer",
        "def": "A standard feedforward neural network is structured into three distinct layer types: the <strong>Input Layer</strong> receives raw features, one or more <strong>Hidden Layers</strong> perform intermediate feature transformations, and the <strong>Output Layer</strong> produces the final prediction.",
        "how": "Input Layer nodes do not perform calculations; they simply pass data forward. Hidden Layers extract progressively abstract representations. The Output Layer formats the prediction into probabilities or continuous values depending on the task.",
        "example": "In our Stage 2 CNN vision network: Input Layer = 224x224x3 image pixels; Hidden Layers = 12 convolutional and pooling layers extracting tumor edge and texture maps; Output Layer = 3 Softmax neurons outputting probabilities for [Benign, Malignant, Inflammation].",
        "viva": "The Input Layer ingests raw features; Hidden Layers extract hierarchical abstract patterns using learned weights; the Output Layer delivers the final task-specific prediction.",
        "remember": "Any neural network with 2 or more hidden layers is classified as a 'Deep' Neural Network."
    },
    {
        "num": 6,
        "title": "Weights and Bias",
        "def": "<strong>Weights (W)</strong> represent the strength and direction of the connection between two neurons (how much influence an input has on the output). <strong>Bias (b)</strong> is an additive constant that shifts the activation curve left or right, allowing the neuron to activate even when all inputs are zero.",
        "how": "In the linear formula y = WX + b: W acts like the slope 'm', determining how steeply output changes with input; b acts like the intercept 'c', setting the baseline firing threshold.",
        "example": "If a patient has zero prior toxicity (x=0), the bias term prevents the neuron from blindly predicting zero risk, allowing baseline risk to reflect general population incidence.",
        "viva": "Weights determine the magnitude and polarity of input influence on the neuron; Bias provides a trainable offset that shifts the activation threshold independently of inputs.",
        "remember": "Weights control the slope; Bias controls the activation threshold intercept."
    },
    {
        "num": 7,
        "title": "Activation Functions",
        "def": "Activation Functions are mathematical equations applied to the weighted sum of a neuron to introduce <strong>non-linearity</strong> into the network. Without non-linear activation functions, stacking 100 neural network layers would simply collapse into a single large linear regression model.",
        "how": "They take the linear value z = WX + b and compress or transform it into non-linear activations.",
        "example": "In our project: We use <strong>ReLU</strong> in all intermediate convolutional layers of our pathology CNN for fast computation; we use <strong>Softmax</strong> in the final layer to output mutually exclusive probabilities for Benign (0.05), Malignant (0.88), and Inflammation (0.07).",
        "viva": "Activation functions introduce non-linearity, enabling networks to learn complex boundaries. The four essential activation functions are:<br>"
               "• <strong>ReLU (Rectified Linear Unit):</strong> f(z) = max(0, z). Returns 0 for negative inputs, linear for positive. Solves vanishing gradient, extremely fast, default for hidden layers.<br>"
               "• <strong>Sigmoid:</strong> f(z) = 1 / (1 + e⁻ᶻ). Squeezes inputs into range (0, 1). Ideal for binary classification outputs; suffers from vanishing gradients in deep layers.<br>"
               "• <strong>Tanh (Hyperbolic Tangent):</strong> f(z) = (eᶻ - e⁻ᶻ) / (eᶻ + e⁻ᶻ). Squeezes inputs into (-1, 1). Zero-centered, preferred over sigmoid in recurrent hidden layers (like LSTM gates).<br>"
               "• <strong>Softmax:</strong> f(zᵢ) = eᶻᶦ / Σ(eᶻʲ). Normalizes a vector of raw logits into a valid probability distribution that sums to 1.0, used for multi-class classification.",
        "remember": "Use <strong>ReLU</strong> for hidden layers; use <strong>Sigmoid</strong> for binary output; use <strong>Softmax</strong> for multi-class output."
    },
    {
        "num": 8,
        "title": "Forward Propagation",
        "def": "Forward Propagation is the initial phase of neural network execution where input data passes through the network from the input layer, across hidden layers, to the output layer to compute the model's prediction.",
        "how": "At each layer l, the linear transformation is computed: Z[l] = W[l]A[l-1] + b[l], followed by non-linear activation: A[l] = σ(Z[l]). This cascades sequentially until the final output ŷ is calculated.",
        "example": "When a new histopathology tile is scanned, forward propagation performs matrix multiplications across convolutional layers to produce the prediction: ŷ = 88% probability of Malignancy.",
        "viva": "Forward propagation is the sequential forward calculation where inputs are multiplied by weights, summed with bias, and transformed by activation functions layer by layer to generate the final prediction.",
        "remember": "Forward pass = Compute Predictions and Loss. No weight updates occur during the forward pass!"
    },
    {
        "num": 9,
        "title": "Loss Function",
        "def": "A Loss Function (or Cost Function) is a mathematical formula that quantifies the difference between the model's predicted output (ŷ) and the actual ground-truth label (y). It measures how 'wrong' the network is.",
        "how": "For classification, we use <strong>Cross-Entropy Loss</strong> (penalizes confident wrong classifications logarithmically). For regression, we use <strong>Mean Squared Error (MSE)</strong>: L = (1/2)(y - ŷ)².",
        "example": "In our project: For 3-class pathology tile classification, we use <strong>Categorical Cross-Entropy Loss</strong>. If the ground truth is Malignant [0, 1, 0] and the network predicts [0.70, 0.20, 0.10], the loss is very high (~1.61), generating strong corrective gradients.",
        "viva": "A loss function measures the discrepancy between predicted and actual values for an individual sample (or batch), producing a single scalar error that guides optimization during backpropagation.",
        "remember": "Classification = Categorical Cross-Entropy or Binary Cross-Entropy. Regression = Mean Squared Error (MSE) or Mean Absolute Error (MAE)."
    },
    {
        "num": 10,
        "title": "Backpropagation",
        "def": "Backpropagation (backward propagation of errors) is the central learning mechanism of neural networks. It calculates the partial derivative (gradient) of the loss function with respect to every single weight and bias in the network by systematically applying the calculus <strong>Chain Rule</strong> backward from the output layer to the input layer.",
        "how": "Starting at the loss L, backprop computes ∂L/∂W[last], then uses the chain rule: ∂L/∂W[l] = (∂L/∂A[l]) * (∂A[l]/∂Z[l]) * (∂Z[l]/∂W[l]), propagating error gradients backward through all hidden layers.",
        "example": "If our CNN misclassifies an inflamed tissue tile as malignant, backpropagation calculates exactly how much each convolutional filter weight contributed to that specific error and directs them how to adjust.",
        "viva": "Backpropagation is the efficient computation of gradients using the calculus chain rule backward through the network, determining the exact rate of change of the loss with respect to every trainable weight and bias.",
        "remember": "Backprop only computes gradients (derivatives); the Optimizer (Gradient Descent / Adam) actually updates the weights!"
    },
    {
        "num": 11,
        "title": "Gradient Descent",
        "def": "Gradient Descent is the optimization algorithm used to train neural networks by iteratively adjusting weights in the opposite direction of the gradient of the loss function to reach the minimum error (bottom of the loss surface).",
        "how": "Weights are updated using the rule: <strong>W_new = W_old - α * (∂L/∂W)</strong>, where α is the learning rate and ∂L/∂W is the gradient computed via backpropagation.",
        "example": "Imagine a blind hiker on a foggy mountain trying to find the lowest valley (minimum loss). Gradient descent steps downhill in the direction of steepest slope at step size α.",
        "viva": "Gradient descent is an iterative first-order optimization algorithm that updates network weights in the direction opposite to the loss gradient to minimize the overall objective cost function.",
        "remember": "Negative sign in formula is critical: <code>W = W - α * ∇L</code>. We subtract because we want to decrease the loss, not increase it."
    },
    {
        "num": 12,
        "title": "Learning Rate",
        "def": "The Learning Rate (α, or eta) is a critical hyperparameter that controls the step size taken in the weight space during each optimization step of gradient descent.",
        "how": "If the learning rate is <strong>too large</strong>, the optimizer will overshoot the optimal minimum and may oscillate or diverge (loss blows up to NaN). If <strong>too small</strong>, training takes an impractically long time and may get trapped in poor local minima or saddle points.",
        "example": "In training our Stage 2 LSTM model on longitudinal biomarker sequences, an initial learning rate of α = 0.1 caused loss instability. Tuning it down to <strong>α = 0.001 with Adam optimizer</strong> enabled smooth, steady convergence over 50 epochs.",
        "viva": "Learning rate is the hyperparameter determining step size along the negative loss gradient. It balances convergence speed against numerical stability and the risk of overshooting the global minimum.",
        "remember": "Typical good default is <code>0.001</code> with adaptive optimizers like Adam, often combined with a learning rate decay scheduler."
    },
    {
        "num": 13,
        "title": "Epoch",
        "def": "An Epoch represents one complete forward and backward pass of the entire training dataset through the neural network.",
        "how": "During one epoch, every single training sample is seen by the model exactly once (typically broken up into mini-batches). Training a network typically requires anywhere from 10 to 200+ epochs until validation loss stabilizes.",
        "example": "Our Stage 2 histopathology training dataset consists of 8,400 training tiles (from 700 training patients). When all 8,400 tiles have been processed and backpropagated through once, 1 epoch is complete.",
        "viva": "An epoch is one complete cycle where the entire training dataset is passed forward and backward through the neural network once.",
        "remember": "Too few epochs &rarr; Underfitting. Too many epochs &rarr; Overfitting. We use Early Stopping to find the ideal number."
    },
    {
        "num": 14,
        "title": "Batch",
        "def": "A Batch is a subset of training samples passed through the neural network simultaneously before computing loss and performing a single weight update.",
        "how": "• <strong>Full Batch (Batch GD):</strong> Uses the entire dataset at once (slow, high GPU memory, smooth gradients).<br>• <strong>Stochastic GD (Batch size = 1):</strong> Updates weights after every single sample (fast, but noisy and erratic updates).<br>• <strong>Mini-Batch GD:</strong> Uses a small subset (typically 16, 32, 64, or 128 samples) — the industry gold standard balancing GPU parallelism and gradient stability.",
        "example": "In our CNN training pipeline, we set <code>batch_size = 32</code>. The GPU processes 32 biopsy tiles in parallel, averages their loss, and updates network weights once per batch.",
        "viva": "A batch is the subset of training examples processed concurrently before the model recalculates gradients and updates its parameters. Mini-batch gradient descent is standard in modern deep learning.",
        "remember": "Batch sizes are conventionally powers of 2 (16, 32, 64, 128) to maximize GPU memory bus architecture efficiency."
    },
    {
        "num": 15,
        "title": "Iteration",
        "def": "An Iteration is the completion of a single weight update step, corresponding to the processing of one mini-batch. The total number of iterations per epoch equals: Total Training Samples / Batch Size.",
        "how": "If a dataset has 8,400 samples and the batch size is 32: Number of iterations per epoch = 8,400 / 32 = 262.5 &rarr; 263 iterations. If trained for 10 epochs, total iterations = 2,630.",
        "example": "During training of our Stage 2 CNN on 8,400 image tiles with batch size 32, the optimizer performs 263 weight updates (iterations) in each epoch.",
        "viva": "An iteration represents a single gradient calculation and weight update cycle on one mini-batch of data.",
        "remember": "1 Epoch = (Total Samples / Batch Size) Iterations."
    },
    {
        "num": 16,
        "title": "Optimizers: SGD vs Adam",
        "def": "Optimizers are mathematical algorithms that update neural network weights based on the computed gradients to minimize the loss function efficiently.",
        "how": "• <strong>SGD (Stochastic Gradient Descent):</strong> Updates weights using the basic formula W = W - α∇L. When enhanced with <strong>Momentum</strong>, it accumulates past velocity like a rolling ball down a hill to push through shallow local minima and dampen oscillations.<br>• <strong>Adam (Adaptive Moment Estimation):</strong> Computes individual adaptive learning rates for each parameter by combining the advantages of AdaGrad (tracking squared gradients: 2nd moment) and RMSProp/Momentum (tracking exponentially decaying moving averages of past gradients: 1st moment).",
        "example": "In our project: SGD with momentum was slow to converge on complex longitudinal biomarker time series. Switching to <strong>Adam (lr=0.001, β₁=0.9, β₂=0.999)</strong> converged 4x faster and reached a lower validation cross-entropy loss.",
        "viva": "SGD updates weights with a fixed learning rate along the gradient; Adam adapts individual learning rates for every parameter by tracking both the first moment (mean) and second moment (uncentered variance) of past gradients, making it the most popular, robust default optimizer.",
        "remember": "Adam = Momentum (direction inertia) + RMSProp (adaptive per-weight step sizes). Start with Adam as your baseline!"
    },
    {
        "num": 17,
        "title": "Training a Neural Network (The 5-Step Loop)",
        "def": "Training a neural network is an iterative algorithmic loop consisting of five systematic operations repeated across multiple batches and epochs until convergence.",
        "how": "The 5 steps are:<br>"
               "1. <strong>Load Mini-batch:</strong> Fetch input tensors X and ground truth targets y.<br>"
               "2. <strong>Forward Pass:</strong> Pass X through all layers to compute predictions ŷ.<br>"
               "3. <strong>Calculate Loss:</strong> Evaluate discrepancy between ŷ and y using the loss function L.<br>"
               "4. <strong>Backward Pass (Backprop):</strong> Compute gradients ∂L/∂W of all trainable weights.<br>"
               "5. <strong>Optimizer Step:</strong> Update weights via optimizer rule (W = W - α∇L) and reset gradients to zero.",
        "example": "In our PyTorch/TensorFlow pipeline: <code>outputs = model(tiles); loss = criterion(outputs, labels); optimizer.zero_grad(); loss.backward(); optimizer.step();</code>.",
        "viva": "Training consists of forward propagation to get predictions, loss computation against ground truth, backpropagation via chain rule to calculate gradients, optimizer weight updates, and repeating across epochs until convergence.",
        "remember": "Always remember <code>optimizer.zero_grad()</code> in PyTorch, otherwise gradients accumulate across batches!"
    },
    {
        "num": 18,
        "title": "Overfitting in Deep Learning",
        "def": "Deep neural networks possess millions of trainable parameters, giving them massive capacity. If not properly controlled, they can easily memorize specific training images, pixel artifacts, or noise instead of generalizable biological concepts, resulting in low training loss but high validation loss.",
        "how": "Observed on a training curve: Training loss continues to decline steadily toward 0, but Validation loss begins to curve upward after a certain epoch. This divergence marks the onset of overfitting.",
        "example": "In Stage 2, our initial CNN achieved 99% accuracy on training biopsy tiles but only 64% on validation tiles because it memorized staining intensity artifacts. Applying Dropout (p=0.4) and Data Augmentation brought validation accuracy up to 82%.",
        "viva": "Overfitting in DL occurs when an over-parameterized network memorizes training noise rather than true underlying features, diagnosed when validation loss begins increasing while training loss keeps decreasing.",
        "remember": "Techniques to fix DL overfitting: 1. Dropout, 2. Data Augmentation, 3. L2 Weight Decay, 4. Early Stopping, 5. Gathering more training data."
    },
    {
        "num": 19,
        "title": "Dropout",
        "def": "Dropout is a powerful regularization technique specifically designed for deep neural networks. During training, it randomly deactivates (sets to zero) a chosen percentage (e.g., 20% to 50%) of neurons in a layer on each forward-backward pass.",
        "how": "By randomly dropping neurons, the network cannot rely on any single neuron or specific co-adapted combination of neurons. Each neuron is forced to learn robust, independent features. At test/inference time, all neurons remain active, and weights are scaled proportionally.",
        "example": "In our Stage 2 CNN dense classification head, we inserted <code>nn.Dropout(p=0.4)</code>. During training, 40% of hidden neurons are randomly shut down in every iteration, completely preventing tissue slide memorization.",
        "viva": "Dropout is a regularization technique that randomly zeroes out a fraction of neurons during training passes to prevent co-adaptation of features, effectively training an ensemble of thinned sub-networks.",
        "remember": "Dropout is active ONLY during training! At inference/test time, dropout is disabled (<code>model.eval()</code>)."
    },
    {
        "num": 20,
        "title": "Regularization in Deep Learning",
        "def": "Regularization techniques modify the learning algorithm to reduce generalization error (test error) without significantly increasing training error.",
        "how": "• <strong>L2 Regularization (Weight Decay):</strong> Adds a penalty term λΣ(W²) to the loss function, discouraging large individual weight values and producing smoother decision boundaries.<br>• <strong>L1 Regularization:</strong> Adds λΣ|W|, encouraging sparsity by driving irrelevant weights to absolute zero.<br>• <strong>Early Stopping:</strong> Monitors validation loss and automatically terminates training when validation loss stops improving for N consecutive epochs (patience).<br>• <strong>Data Augmentation:</strong> Artificially expands the training dataset by applying random label-preserving transformations (rotations, flips, crops).",
        "example": "In our pathology tile pipeline: We applied random horizontal/vertical flips and ±15° rotations to 224x224 biopsy tiles (Data Augmentation) + Early Stopping with patience=7 epochs.",
        "viva": "Regularization prevents overfitting in DL via L2 weight decay (penalizing large weights), Dropout (random neuron muting), Early Stopping (halting on validation plateau), and Data Augmentation (synthesizing diverse inputs).",
        "remember": "Early stopping with patience saves training time, computes the best checkpoint, and prevents late-epoch overfitting automatically."
    },
    {
        "num": 21,
        "title": "CNN (Convolutional Neural Network)",
        "def": "A Convolutional Neural Network (CNN) is a deep learning architecture specifically engineered for grid-structured spatial data like 2D digital images. Instead of using fully connected layers that flatten images and lose spatial structure, CNNs use sliding mathematical filters (kernels) to capture localized visual features.",
        "how": "A CNN consists of three primary layer types:<br>"
               "1. <strong>Convolutional Layer:</strong> Slides small learnable filters (e.g., 3x3 kernels) over the input image to compute element-wise dot products, generating <strong>Feature Maps</strong> that detect edges, textures, and cellular morphology.<br>"
               "2. <strong>Activation Layer (ReLU):</strong> Introduces non-linearity to feature maps.<br>"
               "3. <strong>Pooling Layer (Max Pooling):</strong> Downsamples spatial dimensions (e.g., 2x2 max pool reduces height and width by half) while retaining the strongest activations, achieving translation invariance and reducing compute.<br>"
               "4. <strong>Dense / Fully-Connected Layer:</strong> Flattens downsampled high-level feature maps into a vector and outputs final classification probabilities via Softmax.",
        "example": "In Stage 2 of our project: We process 224x224x3 H&E biopsy tiles through a CNN with 3 Conv-BatchNorm-ReLU-MaxPool blocks followed by a Dense layer to classify tiles into [Benign, Malignant, Inflammation].",
        "viva": "CNNs are neural networks designed for computer vision that use parameter-sharing convolutional filters to extract spatial feature hierarchies, combined with pooling layers for dimensional downsampling and spatial invariance.",
        "remember": "Key advantages of CNNs: <strong>Parameter Sharing</strong> (same filter used across entire image) and <strong>Sparsity of Connections</strong> (local receptive fields)."
    },
    {
        "num": 22,
        "title": "RNN (Recurrent Neural Network)",
        "def": "A Recurrent Neural Network (RNN) is a neural network architecture designed for processing sequential data (time-series, text, temporal signals). Unlike feedforward networks where inputs are assumed independent, an RNN maintains an internal memory (hidden state) that loops back into itself.",
        "how": "At each time step t, the RNN takes current input xₜ and previous hidden state hₜ₋₁ to compute the new hidden state: hₜ = tanh(Wₕₕ hₜ₋₁ + Wₓₕ xₜ + b). Information from earlier time steps persists across the sequence.",
        "example": "In oncology, a patient visits the hospital across 6 consecutive cycles. An RNN passes the hidden state from Cycle 1 through Cycle 6 to track the cumulative physiological impact of repeated chemotherapy infusions.",
        "viva": "An RNN is a sequential neural network containing recurrent feedback loops that pass hidden state representations from previous time steps forward, allowing the network to maintain memory over temporal sequences.",
        "remember": "Standard Vanilla RNNs suffer severely from the <strong>Vanishing Gradient Problem</strong>, making them incapable of learning dependencies over more than 10–15 time steps."
    },
    {
        "num": 23,
        "title": "LSTM (Long Short-Term Memory)",
        "def": "Long Short-Term Memory (LSTM) is an advanced variant of RNN specifically engineered to overcome the vanishing gradient problem. It introduces a dedicated memory pipeline called the <strong>Cell State (Cₜ)</strong> and regulates the flow of information using three specialized mathematical gates.",
        "how": "An LSTM cell operates using three gates:<br>"
               "1. <strong>Forget Gate (fₜ = σ):</strong> Decides what old information from cell state Cₜ₋₁ should be discarded.<br>"
               "2. <strong>Input Gate (iₜ = σ & C̃ₜ = tanh):</strong> Decides what new information from input xₜ and hidden state hₜ₋₁ should be stored in the cell state.<br>"
               "3. <strong>Output Gate (oₜ = σ):</strong> Decides what parts of the updated cell state Cₜ should be emitted as hidden state output hₜ.",
        "example": "In Stage 2 of our project, we use an LSTM to analyze 180-day longitudinal biomarker sequences (historical Days 0–90 inputs) to forecast whether a patient's circulating tumor DNA (ctDNA) will spike 30 days into the future.",
        "viva": "LSTM is a recurrent network architecture that solves the vanishing gradient problem using a constant error carousel (cell state) controlled by three multiplicative gates: Forget, Input, and Output gates.",
        "remember": "The 3 Gates: <strong>Forget Gate</strong> (what to drop), <strong>Input Gate</strong> (what to add), <strong>Output Gate</strong> (what to emit). Gate activations use Sigmoid (0 to 1); candidate states use Tanh (-1 to 1)."
    },
    {
        "num": 24,
        "title": "Transformers",
        "def": "The Transformer is a state-of-the-art deep learning architecture that dispenses entirely with recurrence and convolutions, relying instead solely on the <strong>Self-Attention Mechanism</strong> to process sequential and contextual data in parallel.",
        "how": "Instead of stepping through tokens one by one like an RNN, Transformers process the entire sequence at once. Self-Attention calculates mathematical affinity scores between all pairs of tokens in the sequence using Query (Q), Key (K), and Value (V) matrices: Attention(Q, K, V) = softmax(QKᵀ / √dₖ) * V. Positional encodings are added to retain sequence order.",
        "example": "In advanced clinical systems: Transformers (like ClinicalBERT or BioBERT) process unstructured clinical notes and longitudinal pathology reports simultaneously to capture subtle distant correlations between clinical symptoms recorded months apart.",
        "viva": "Transformers are attention-based neural architectures that process sequential inputs entirely in parallel, computing global contextual relationships across all tokens simultaneously using multi-head self-attention without recurrent bottlenecks.",
        "remember": "Transformers power modern LLMs (GPT-4, Gemini) and are increasingly used in multi-modal healthcare to fuse text, images, and tabular streams."
    },
    {
        "num": 25,
        "title": "Applications of Deep Learning in Healthcare & Our Project",
        "def": "Deep Learning has transformed modern medicine by enabling automated medical image interpretation, genomic sequence modeling, drug discovery, and predictive clinical trajectory forecasting.",
        "how": "In our Precision Oncology project, Deep Learning is applied in two integrated modalities:<br>"
               "1. <strong>Vision (CNN):</strong> Automated digital histopathology tile classification (Benign, Malignant, Inflammation) from high-resolution biopsy scans.<br>"
               "2. <strong>Temporal Forecasting (LSTM/Transformer):</strong> Longitudinal ctDNA progression forecasting from serial liquid biopsy measurements across treatment cycles.<br>"
               "3. <strong>Late Fusion:</strong> Combining vision scores and temporal risk scores into a unified clinical alert API for oncologist decision support.",
        "example": "When an oncologist orders a biopsy and liquid blood draw, our Stage 2 DL pipeline automatically flags an aggressive malignant tissue tile (CNN confidence = 94%) and predicts an impending ctDNA surge (LSTM 30-day forecast), triggering an urgent clinical alert.",
        "viva": "In our project, DL powers automated histopathology tissue classification via CNNs and multi-cycle longitudinal ctDNA biomarker forecasting via LSTMs, fused into a clinical decision support API to prevent adverse patient outcomes.",
        "remember": "Multimodal Deep Learning = Combining different data types (e.g., visual pathology + temporal blood biomarkers) to achieve superior diagnostic safety over any single data stream."
    },
    {
        "num": 26,
        "title": "Deep Learning Evaluation Metrics",
        "def": "Deep learning models are evaluated using both optimization metrics (loss functions tracked during training) and real-world task metrics (classification and regression benchmarks computed on the locked test set).",
        "how": "Key metrics include: Categorical Cross-Entropy Loss, Area Under the ROC Curve (AUC-ROC), Precision-Recall AUC (PR-AUC), Top-1 Accuracy, and for sequence forecasting, Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).",
        "example": "In Stage 2: Our CNN histopathology classifier achieved <strong>AUC-ROC = 0.89</strong> and <strong>Macro F1 = 0.82</strong> across 1,800 locked test tiles; our LSTM ctDNA forecaster achieved <strong>RMSE = 0.18 ng/mL</strong> on 30-day progression predictions.",
        "viva": "DL models are evaluated during training by tracking training vs validation loss curves to monitor overfitting, and on test data using task-specific metrics including AUC-ROC, Macro F1, and RMSE.",
        "remember": "Always track both Training and Validation Loss per epoch; if training loss falls but validation loss rises, the deep network is overfitting."
    },
    {
        "num": 27,
        "title": "ML vs DL Comparison",
        "def": "Understanding the fundamental boundary between Machine Learning and Deep Learning is essential for any data science viva.",
        "how": "See the complete comparison table in Part 4 summarizing Dataset Requirements, Feature Engineering, Hardware, Interpretability, Training Time, and Clinical Project Application.",
        "example": "In our project: Stage 1 ML was chosen for structured tabular EHR patient encounters (1,750 records) using LightGBM; Stage 2 DL was chosen for unstructured 224x224 RGB image tiles and multi-step temporal sequences using CNNs and LSTMs.",
        "viva": "ML is best suited for structured tabular data with explicit feature engineering on CPUs; DL excels on unstructured high-dimensional data (images, audio, sequences) with automatic feature representation learning on GPUs.",
        "remember": "Do not use Deep Learning when a simple Random Forest or LightGBM model solves the tabular problem with higher interpretability and less computational cost!"
    }
]
