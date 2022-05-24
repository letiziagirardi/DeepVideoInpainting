# Intership in Video Inpainting &amp; post-processing

This repository contais all steps of my intership took place at the University of Trento. 

## Short description of what I do and why

Video Inpainting refers to a field of Computer Vision that aims to remove objects or restore missing or tainted regions present in a video sequence by utilizing spatial and temporal information from neighbouring scenes. The overriding objective is to generate an inpainted area that is merged seamlessly into the video so that visual coherence is maintained throughout and no distortion in the affected area is observable to the human eye when the video is played as a sequence.

It is important at this stage to distinguish video and image inpainting from the related field of “texture synthesis”.

Recently, we have seen rapid progress in image in-painting activity, however video in-painting remains at an early stage as it is still little explored.
The ideas most commonly used in in-painting, in fact, are not applicable to the problem of video in-painting: the networks used in the first case have limited or directional temporal receptive fields that appear to be incompatible for in-painting video. 
The key point to note is that image inpainting techniques only consider spatial information, and completely neglect the significant temporal component present in video sequences. We know that, in the field of video completion, we need to fill in the missing regions with content that is consistent over time, since, when you remove an object in a video, the region occluded by the object may be visible in other frames. Therefore, filling the region of the frame being analyzed without considering the original content in the other frames will break the temporal coherence.

## Preliminary knowledge

The network HPFCN has been tested on two types of data inpainted: 
* **MODEL A** has been obtained from training the network on data inpainted by using just one of the three techniques GMCNN, OPN, STTN
* **MODEL B** has been obtained from training the network on data inpainted by using just one of the three techniques, post-processed with TCN.


## ✅ STEP #1

During the first step, the work could be summarize as follows:
* training of HPFCN on data inpainted with a specific technique between GMCNN, OPN and STTN; in follow paragraph we will called it model (a);
* testing model a on data inpainted with both GMCNN and OPN and STTN;
* testing of model a on data inpainted with both GMCNN and OPN and STTN, post-processed with TCN;
* training of HPFCN on data inpainted with a specific technique between GMCNN, OPN and STTN which has been processed with TCN too; in follow paragraph we will called it model (b);
* testing model bon data inpainted with both GMCNN and OPN and STTN;
* testing of model b on data inpainted with both GMCNN and OPN and STTN, post-processed with TCN;

## ✅ STEP #2

During this step, I analyzed some experiments in order to classify the best technique of inpainting between GMCNN, OPN and STTN. Firstly, in a qualitative way. Secondly in a quantitative way.

### First experiment consisted of analyzing the differences between prediction maps representing the first frame processed of the video obtained by training HPFCN on data inpainted with GMCNN and testinig them on both techniques GMCNN, OPN and STTN. 
   RESULTS: It’s evident that each frame obtained training HPFCN with data manipulated according to the testing of model a - model obtain by the training of HPFCN on data inpainted with GMCNN technique - on the basic techniques GMCNN, OPN and STTN, is cleaner than the respective frame obtained by the testing of the same model on manipulated data with TCN too. 
  This denotes the nature of the neural network TCN. The proposed network aims to enforce temporal consistency on videos by generating output frames that are temporally consistent with respect to the previous frames. Due to this the network has to set the first frame of the output video equal to the first frame taken in input, giving first prediction maps imprecise.
  
### Second experiment consisted of analyzing each output frame obtained from the different testing of model a on the several techniques GMCNN, OPN and STTN. This part aims to compare the performance of each technique without processing any data with TCN.
  RESULTS: From these eexperiments I observed that the output of the HPFCN produces better localization maps on the same technique it was trained with. For instance, if the HPFCN was trained on data inpainted with GMCNN, the best localization map is on the training of it with GMCNN. It is demonstrated with Figure A, B and C.

### Third experiment consisted of showing how TCN worsens the completion of the video.
  RESULTS: From these experiments, we can claim that every manipulation with TCN is not able to reproduce exactly the object and therefore data manipulated without TCN are preferable for this task.
Frames are not clear: the object is not perfectly recognizable and on the first examination, the image has lots of white areas which represent external objects.

-----------------------------

To measure the performance of all these tests, I observed them in terms of **F1-score**.
In statistical analysis, the F-score is a measure of a test’s accuracy. It is calculated from the precision and recall of the test, where the precision is the number of true positive results divided by the number of all positive results, and the recall is the number of true positive results divided by the number of all samples that should have been identified as positive. 

Namely, we are interested in the model with the _highest validation F1-score_.

## ✅ STEP #3

Studing how videos inpainted and post-processed with TCN look like.

From the results of the previous experiments it was clear the adversarial effect that deep learning based post-processing techniques have on multimedia forensics tasks as the one of this report. Knowing that TCN was implemented, we choose to start to analyse if any artefacts were visible in the Fast Fourier Transform (FFT) Domain.

We computed the FFT of frames processed and not with TCN by applying the FFT to each channel of every frame and computing their real part. Thus, the real parts of the FFT compute per-every channel have been concatenated so that to obtain a 3 channel FFT transform as the ones of Figure 4. From the results of Figure 4 we can infer few things:
• when applied TCN introduced distinguishable artefacts on the FFT high frequencies;
• these artefacts are less prominent in presence of highly textured frames;

In addition to this, when we isolate the high frequencies at the center of the FFT transforms with a circumference of radii equals at 50 pixels, we can observe in the residuals that TCN reduced consistently the amount of high frequencies sources (edge, textures, etc.) and that, the distribution of the variances of these residuals change consistently. 

Using the obtained variances we defined a score with which we built a ROC: a graphic scheme for binary classification. The ROC represents, for each point of the curves, the probability to correctly classify a TCN post-processed frame given a value of False Positives. 
  RESULTS: We are already able to distinguish TCN processed frames from the not processed ones with a TPR≈0.75 and a FPR≈0.05. 


## WORKING ON STEP #4

At this point we try to train the network HPFCN with new residual obtained in step #3. The residuals consist of just high frequencies. We expect that the network, working on high frequencies, is able to detect the removed object in a better way. 

