# Counterfactual-ultrasoundAI
Counterfactual Ultrasound Anti-Interference Self-Supervised Network for B-mode Ultrasound Tongue Extraction


**Overview of our framework.** **Grid Dropout (GD):** Dropout is applied to specific pixels in a grid-like fashion. **Center Dropout (CD):**
A large contiguous area is specified for pixel dropout. **Center Filter (CF):** After selecting the central region, basic mean filtering is applied.
**Random Shuffle (RS):** Regions are randomly selected for shuffling pixel combinations in blocks.

![SCDA](https://github.com/inexhaustible419/Counterfactual-ultrasoundAI/assets/145007561/ac97a37a-47ff-4e9c-a778-957b4f6eb75f)



**Typical sample performance in our method ablation study.** **Original** is the direct output of the backbone network; **+SSL** adds self
supervised pre training; **+RS&GD** adds specific data augmentation based on **+SSL**.
![1-Result](https://github.com/inexhaustible419/Counterfactual-ultrasoundAI/assets/145007561/b4467053-a197-41e9-9830-e3e5839dbeca)


**Supported by the Open Fund of Science and Technology on Parallel and Distributed Processing Laboratory ( PDL )**
