# Online Projector Deblurring Using a Convolutional Neural Network (IEEE TVCG 2022, IEEE VR 2022)
![result](https://user-images.githubusercontent.com/40446914/154205594-2c8e4233-b0ce-4f95-b58b-b56dd2ff90c2.png)
- Paper: https://ieeexplore.ieee.org/document/9714047
- Video: https://youtu.be/MSCj2IIZevw
## Highlights
- This repositry describes how to use the networks used in our paper (i.e., DefocusNet, LuminanceNet, and CompensationNet).
- The projection image is assumed to be grayscale with a height and width of 256.
- If you want to project color images, use a network for each channel (i.e., red, green, and blue).
## Network architecture
![architecture](https://user-images.githubusercontent.com/40446914/154384020-90a5e4a9-1dda-4107-8b0b-6757c2cb6dc4.png)
## Implementation
### Requirements
- Python 3
- opencv-python 3.4.2.16
- tensorflow 2.3.1
- tensorflow-probability 0.11.0
- Keras 2.4.0
### How to install
1. Clone this repository.

        git clone https://github.com/kagechan5/Online-Projector-Deblurring-Using-a-Convolutional-Neural-Network.git
        cd Online-Projector-Deblurring-Using-a-Convolutional-Neural-Network

2. Install required packages.
        
        pip install -r requirements.txt

3. Install our package from PyPI.

        pip install onlineProdebnet
### How to use networks
Note that all of the following processes are assumed to be written in source files located same hierarchy as weight folder.
1. Load models.

        import onlineProdebnet
        imgHeight, imgWidth = 256, 256
        DefocusNet = onlineProdebnet.DefocusNet.loadModel((imgHeight,imgWidth,1),"./weight/defocus_weight.h5")
        LuminanceNet = onlineProdebnet.LuminanceNet.loadModel((imgHeight,imgWidth,1),"./weight/luminance_weight.h5")
        CompensationNet = onlineProdebnet.CompensationNet.loadModel((imgHeight,imgWidth,1),"./weight/compensation_weight.h5")

2. Estimate defocus blur map and luminance attenuation map.
 
        import numpy as np
        initialMap = np.zeros((1,imgHeight,imgWidth,1),dtype="float32")
        input2DefocusNet = np.concatenate([preProjectionImage,preProjectedResult,initialMap],-1)
        input2LuminanceNet = np.concatenate([preProjectionImage,preProjectedResult,initialMap],-1)
        defocusMap = DefocusNet.predict(input2DefocusNet)
        luminanceMap = LuminanceNet.predict(input2LuminanceNet)

      #### Notes
      - This process is only done after projecting the target image of first frame.
      - "preProjectionImage" is the target image of first frame.
      - "preProjectedResult" is the projected result of the target image of first frame.
      - The shape of "preProjectionImage" and "preProjectedResult" must be (1, imgHeight, imgWidth, 1), the data type must be float32, and the values must be normalized to 0~1.

3. Generate the compensation image.
        
        input2CompensationNet = np.concatenate([targetImage,defocusMap,luminanceMap],-1)
        compensationImage = CompensationNet.predict(input2CompensationNet)

      #### Notes
      - "targetImage" is the target image of t frame (t >= 2).
      -  The shape of "targetImage" must be (1, imgHeight, imgWidth, 1), the data type must be float32, and the values must be normalized to 0~1.

4. Estimate defocus blur map and luminance attenuation map.

        input2DefocusNet = np.concatenate([preProjectionImage,preProjectedResult,defocusMap],-1)
        input2LuminanceNet = np.concatenate([preProjectionImage,preProjectedResult,luminanceMap],-1)
        defocusMap = DefocusNet.predict(input2DefocusNet)
        luminanceMap = LuminanceNet.predict(input2LuminanceNet)
        
      #### Notes
      - "preProjectionImage" is the compensation image of t frame (= "compensationImage" generated in step 3).
      - "preProjectedResult" is the projected result of the compensation image of t frame.
      - The shape of "preProjectionImage" and "preProjectedResult" must be (1, imgHeight, imgWidth, 1), the data type must be float32, and the values must be normalized to 0~1.

5. Add 1 to t and repeat steps 3 and 4.

## License
This software is licensed under the [MIT](LICENSE) license. 
<br>Copyright © 2022, Yuta Kageyama
        
 
        
        
    
        
  
