****************************** SPECIFICATION ****************************

Model = 13

Variables:
    
    z = SkillRelevance
    m1 = MediationSkills
    m2 = ModerationSkills
    y = Success
    Cons = Cons
    x = Effort
    w = Motivation

Sample size:

    1000

Bootstrapping information for indirect effects:
    
    Final number of bootstrap samples: 5000
    Number of samples discarded due to convergence issues: 0

***************************** OUTCOME MODELS ****************************

Outcome = Success 

OLS Regression Summary

| R²     | Adj. R² | MSE    | F         | df1 | df2 | p-value |
|--------|---------|--------|-----------|-----|-----|---------|
| 0.9166 | 0.9161  | 1.0300 | 2186.2053 | 5   | 994 | 0.0000  |

Coefficients

|                   | Coeff  | se     | t       | p      | LLCI   | ULCI   |
|-------------------|--------|--------|---------|--------|--------|--------|
| Cons              | 1.0406 | 0.0816 | 12.7588 | 0.0000 | 0.8807 | 1.2005 |
| Effort            | 1.0297 | 0.0530 | 19.4124 | 0.0000 | 0.9257 | 1.1337 |
| Motivation        | 0.9737 | 0.0482 | 20.2208 | 0.0000 | 0.8793 | 1.0681 |
| Effort\*Motivation | 1.0372 | 0.0394 | 26.3276 | 0.0000 | 0.9600 | 1.1144 |
| MediationSkills   | 0.9830 | 0.0231 | 42.5563 | 0.0000 | 0.9377 | 1.0283 |
| ModerationSkills  | 0.9852 | 0.0234 | 42.0542 | 0.0000 | 0.9393 | 1.0312 |

-------------------------------------------------------------------------

Outcome = MediationSkills 

OLS Regression Summary

| R²     | Adj. R² | MSE    | F         | df1 | df2 | p-value |
|--------|---------|--------|-----------|-----|-----|---------|
| 0.9843 | 0.9841  | 1.0009 | 8864.2139 | 7   | 992 | 0.0000  |

Coefficients

|                                  | Coeff   |  se    | t        | p      |   LLCI  | ULCI    |
|------------------------------------|--------|--------|---------|--------|--------|--------|
| Cons                               | 1.0978 | 0.0939 | 11.6877 | 0.0000 | 0.9137 | 1.2819 |
| Effort                             | 0.9466 | 0.0654 | 14.4710 | 0.0000 | 0.8184 | 1.0748 |
| Motivation                         | 0.9653 | 0.0706 | 13.6673 | 0.0000 | 0.8269 | 1.1038 |
| SkillRelevance                     | 0.9879 | 0.0619 | 15.9599 | 0.0000 | 0.8666 | 1.1093 |
| Effort\*Motivation                 | 1.0014 | 0.0473 | 21.1905 | 0.0000 | 0.9088 | 1.0941 |
| Effort\*SkillRelevance             | 1.0131 | 0.0448 | 22.6254 | 0.0000 | 0.9253 | 1.1008 |
| Motivation\*SkillRelevance         | 1.0016 | 0.0451 | 22.2154 | 0.0000 | 0.9132 | 1.0899 |
| Effort\*Motivation\*SkillRelevance | 1.0038 | 0.0299 | 33.5929 | 0.0000 | 0.9453 | 1.0624 |

-------------------------------------------------------------------------

Outcome = ModerationSkills

OLS Regression Summary

| R²     | Adj. R² | MSE    | F         | df1 | df2 | p-value |
|--------|---------|--------|-----------|-----|-----|---------|
| 0.9841 | 0.9839  | 1.0067 | 8750.9642 | 7   | 992 | 0.0000  |



Coefficients

|                                  | Coeff   |  se    | t        | p      |   LLCI  | ULCI    |
|----------------------------------|---------|--------|----------|--------|---------|---------|
| Cons                             | 0.8708  | 0.0752 | 11.5839  | 0.0000 | 0.7235  | 1.0182  |
| Effort                           | -1.0068 | 0.0550 | -18.3189 | 0.0000 | -1.1145 | -0.8991 |
| Motivation                       | -1.0411 | 0.0489 | -21.2928 | 0.0000 | -1.1369 | -0.9453 |
| SkillRelevance                   | -0.9288 | 0.0553 | -16.7860 | 0.0000 | -1.0372 | -0.8203 |
| Effort\*Motivation                | -0.9637 | 0.0374 | -25.7996 | 0.0000 | -1.0369 | -0.8904 |
| Effort\*SkillRelevance            | -0.9786 | 0.0372 | -26.3307 | 0.0000 | -1.0515 | -0.9058 |
| Motivation\*SkillRelevance        | -0.9796 | 0.0389 | -25.1996 | 0.0000 | -1.0558 | -0.9034 |
| Effort\*Motivation\*SkillRelevance | -1.0238 | 0.0270 | -37.9556 | 0.0000 | -1.0767 | -0.9709 |

-------------------------------------------------------------------------


********************** DIRECT AND INDIRECT EFFECTS **********************

Conditional direct effect(s) of Effort on Success at values of the moderator(s):

| Motivation | Effect | SE     | t       | p      | LLCI   | ULCI   |
|------------|--------|--------|---------|--------|--------|--------|
| 0.0124     | 1.0426 | 0.0528 | 19.7646 | 0.0000 | 0.9392 | 1.1460 |
| 1.0131     | 2.0805 | 0.0433 | 48.0769 | 0.0000 | 1.9956 | 2.1653 |
| 2.0137     | 3.1183 | 0.0638 | 48.8759 | 0.0000 | 2.9933 | 3.2434 |

Conditional indirect effect(s) of Effort on Success at values of the moderator(s):

| Mediator         | SkillRelevance | Motivation | Effect  | Boot SE | BootLLCI | BootULCI      |
|------------------|----------------|------------|---------|---------|----------|---------------|
| MediationSkills  | -0.0067        | 0.0124     | 0.9359  | 0.0674  | 0.8084   | 1.0743        |
| MediationSkills  | -0.0067        | 1.0131     | 1.9143  | 0.0653  | 1.7858   | 2.0437        |
| MediationSkills  | -0.0067        | 2.0137     | 2.8926  | 0.0957  | 2.7103   | 3.0783        |
| MediationSkills  | 1.0041         | 0.0124     | 1.9550  | 0.0622  | 1.8357   | 2.0798        |
| MediationSkills  | 1.0041         | 1.0131     | 3.9314  | 0.0985  | 3.7362   | 4.1193        |
| MediationSkills  | 1.0041         | 2.0137     | 5.9078  | 0.1477  | 5.6216   | 6.1960        |
| MediationSkills  | 2.0150         | 0.0124     | 2.9740  | 0.0900  | 2.8005   | 3.1517        |
| MediationSkills  | 2.0150         | 1.0131     | 5.9485  | 0.1471  | 5.6597   | 6.2372        |
| MediationSkills  | 2.0150         | 2.0137     | 8.9230  | 0.2197  | 8.4938   | 9.3538        |
| ModerationSkills | -0.0067        | 0.0124     | -0.9972 | 0.0603  | -1.1166  | -0.8801       |
| ModerationSkills | -0.0067        | 1.0131     | -1.9404 | 0.0586  | -2.0557  | -1.8243       |
| ModerationSkills | -0.0067        | 2.0137     | -2.8836 | 0.0812  | -3.0447  | -2.7289       |
| ModerationSkills | 1.0041         | 0.0124     | -1.9845 | 0.0641  | -2.1118  | -1.8617       |
| ModerationSkills | 1.0041         | 1.0131     | -3.9480 | 0.1000  | -4.1378  | -3.7488       |
| ModerationSkills | 1.0041         | 2.0137     | -5.9114 | 0.1479  | -6.2010  | -5.6206       |
| ModerationSkills | 2.0150         | 0.0124     | -2.9719 | 0.0927  | -3.1537  | -2.7917       |
| ModerationSkills | 2.0150         | 1.0131     | -5.9556 | 0.1538  | -6.2513  | -5.6467       |
| ModerationSkills | 2.0150         | 2.0137     | -8.9393 | 0.2306  | -9.3930  | -8.4850</pre> |

**************** INDEX OF MODERATED MODERATED MEDIATION ******************

| Mediator         | Index   | Boot SE | BootLLCI | BootULCI |
|------------------|---------|---------|----------|----------|
| MediationSkills  | 0.9867  | 0.0566  | -0.7769  | -0.7769  |
| ModerationSkills | -1.0087 | 0.0531  | 0.8091   | 0.8091   |

**************** INDEX OF CONDITIONAL MODERATED MEDIATION ******************

| Focal Mod      | Mediator         | Other Mod At | Index   | Boot SE | Boot LLCI | Boot ULCI     |
|----------------|------------------|--------------|---------|---------|-----------|---------------|
| Motivation     | MediationSkills  | -0.0067      | 0.9777  | 0.0508  | 0.8812    | 1.0805        |
| Motivation     | MediationSkills  | 1.0041       | 1.9752  | 0.0560  | 1.8665    | 2.0861        |
| Motivation     | MediationSkills  | 2.0150       | 2.9726  | 0.0809  | 2.8193    | 3.1337        |
| Motivation     | ModerationSkills | -0.0067      | -0.9426 | 0.0410  | -1.0294   | -0.8667       |
| Motivation     | ModerationSkills | 1.0041       | -1.9622 | 0.0546  | -2.0764   | -1.8610       |
| Motivation     | ModerationSkills | 2.0150       | -2.9819 | 0.0849  | -3.1560   | -2.8215       |
| SkillRelevance | MediationSkills  | 0.0124       | 0.9967  | 0.0507  | 0.9008    | 1.0997        |
| SkillRelevance | MediationSkills  | 1.0131       | 1.9840  | 0.0562  | 1.8748    | 2.0952        |
| SkillRelevance | MediationSkills  | 2.0137       | 2.9714  | 0.0809  | 2.8181    | 3.1324        |
| SkillRelevance | ModerationSkills | 0.0124       | -0.9620 | 0.0410  | -1.0488   | -0.8864       |
| SkillRelevance | ModerationSkills | 1.0131       | -1.9713 | 0.0549  | -2.0854   | -1.8695       |
| SkillRelevance | ModerationSkills | 2.0137       | -2.9806 | 0.0849  | -3.1546   | -2.8203       |
