## Experiment v_1
--based **KB-GAT(1-hop)**
### modify
1. The new embedding of the entity e_i is **the sum of each triple representation** weighted by their attention values 
	![](http://latex.codecogs.com/svg.latex?{e_j||e_i||g_k})
2. Relations lose their initial embedding information
3. 对关系r的更新用的GAT,t和h直接拼接然后和关系r求注意力再聚合，聚合的是W(h||t||r)
	
注：未加入hyper对实体关系作交互
### Effect
![image](https://github.com/Ckeai/model/blob/modify_model/result.png)
