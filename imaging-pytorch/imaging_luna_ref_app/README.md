##### PyTorch for Medical Imaging Applications
- Deep Learning in Applications 
- Monai
- Case Studies 


##### [Medical imaging] PyTorch Reference Application to detect lung cancer
CT scans are stored on disks, they contain density information, and the models consume various sub-slices of those 3D arrays. There is a multi-step process to go over the entire chest CT scan to give a patient a lung cancer diagnosis. The 3D CT scans product of the CT instance contains a full 3D scan. We combine that with a module that performs segmentation (flagging voxels of interest), and then group the interesting voxels into small lumps in the search for candidate 'nodules. The 'nodule' locations are combined back with CT voxel data to produce nodule candidates which are examined by nodule classifier whether the are actually nodules in the first place. 
The end-to-end implementation is the process of taking a CT scan and determining whether the patient has a malignant tumor.  With CT alone it might not be apparent therefore we take on more deep dive, and look into the data for each of the individual scans, per-module classifications that can then be combined into a whole patient diagnosis. 
Classify candidate modules as actual modules or non-modules using 3D convolution.
Diagnose the patient using the combined per-nodule classifications.
Similar to the nodule classifier, attempt to determine whether the nodule is benign or malignant based on the imagining data alone.Taking the simple maximum of the per-tumor malignancy predictions, as one tumor needs to be malignant for a patient to have cancer.
<table align='top'>
<td> <img width="390" alt="image" src="https://user-images.githubusercontent.com/67139134/235386508-79985a85-7d10-4434-ae0a-b49c9515e7e5.png"> </td>
<td> <img width="380" alt="image" src="https://user-images.githubusercontent.com/67139134/235386079-8239cdbc-f293-46f2-9bd1-433b84c3d54c.png"> </td>
<td> <img width="205" alt="image" src="https://user-images.githubusercontent.com/67139134/235385410-0c909a69-1511-405a-ae55-745450f3e3b3.png"> </td> 
<td> <img width="200" alt="image" src="https://user-images.githubusercontent.com/67139134/235385050-652e44a1-c929-4a5a-bfeb-1b467658a4f2.png"> </td>
</table>



##### [Medical imaging] Challenge to predict FVC decline based on high-res CT scan. 
Predict relative or absolute FVC decline given CT(1) and FVC(t1). Perform initial data, sanity check and performance of baseline algorithms. 
FEV1 and FVC: The FEV1/FVC ratio, also called Tiffeneau-Pinelli index,[1] is a calculated ratio used in the diagnosis of obstructive and restrictive lung disease.[2][3] It represents the proportion of a person's vital capacity that they are able to expire in the first second of forced expiration (FEV1) to the full, forced vital capacity (FVC).[4] The result of this ratio is expressed as FEV1%.
The evaluation metric is a modified version of Laplace Log Likelihood. In medical applications, it is useful to evaluate a model's confidence in its decisions. Accordingly, the metric is designed to reflect both the accuracy and certainty of each prediction. For each true FVC measurement, we predict both an FVC and a confidence measure (standard deviation σ).
