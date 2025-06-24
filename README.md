# causal_discovery_CRC
Install the required packages: 
```
pip install -r requirements.txt
```

Crucially, set your OpenAI API key as an environment variable:
```
export OPENAI_API_KEY="your-api-key-here"
```  
Run the calibration:

Open your terminal in the project's root directory.
Execute the main script:
```
python calibrate_causality_model.py
```
You can customize the run with arguments, for example, to set a different risk level:
```
python calibrate_causality_model.py --alpha 0.05 --model_name gpt-3.5-turbo
```

The script will first run the OpenAI model over the calibration data (and save the results), then use those results to calculate and output the final lambda value.
