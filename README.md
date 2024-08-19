# CP-PLL
<h1>CP-PLL: Conformal Prediction for Partially Labeled Learning</h1>

<h2>Overview</h2>

<p>CP-PLL (Conformal Prediction for Partially Labeled Learning) is a powerful framework designed to provide reliable predictions in scenarios where only partial labels are available. This repository contains scripts and resources to preprocess data, run models, generate scores, and visualize results.</p>

<h2>Project Structure</h2>
<ul>
<li><code>models/</code> - Directory containing model architectures for generating scores.</li>
<li><code>npy_generator.py</code> - Script to generate npy files from each datapoint.</li>
<li><code>requirements.txt</code> - Python dependencies.</li>
<li><code>results.py</code> - Script to generate the partial label prediction set size.</li>
</ul>

<h2>How to Run</h2>

<h3>Step 1: Generate Scores</h3>

<p>Run the models inside the <code>models/</code> directory to generate scores. Instructions for running the models are provided inside the respective folders for all five models.</p>

<h3>Step 2: Generate NPY Files</h3>

<p>Use the following command to run the <code>npy_generator.py</code> script and generate npy values from each datapoint:</p>

<pre><code>python npy_generator.py --dataset cifar10 --base_path /Users/nitinbisht/PycharmProjects/ct-pll/exp_alice/proden/balance/balanced_/text_data
</code></pre>

<h3>Step 3: Generate Prediction Set Sizes</h3>

<p>Run the <code>results.py</code> script to generate the partial label prediction set size:</p>

<pre><code>python results.py --base_path /Users/ct-pll/table_cifar_100_lt --epsilon 0.1 --partial_rate 0.2
</code></pre>

<p>The generated prediction sets can then be used to create plots.</p>
