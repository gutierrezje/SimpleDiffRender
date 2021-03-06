# SimpleDiffRender.jl

A Julia port of [Tzu-Mao Li's diffrender_tutorials](https://github.com/BachiLi/diffrender_tutorials)  
It is a small 2D renderer that incorporates ideas from [Differentiable Monte Carlo Ray Tracing through Edge Sampling [Li et al.], 2018](https://cseweb.ucsd.edu/~tzli/diffrt/). An in depth code walkthrough can be found in the SIGGRAPH 2020 course on [Physics-Based Differentiable Rendering](https://shuangz.com/courses/pbdr-course-sg20/).  
Images shown are of a conventional rendering and renderings of screen coordinate derivatives, captured with the edge sampling technique.  
![render](images/render.png)  
| Derivatives in -x and -y direction | Derivatives in x and y direction |
:-------------------------:|:-------------------------:
| ![dx_neg](images/dx_neg.png) | ![dx_pos](images/dx_pos.png) |
| ![dy_neg](images/dy_neg.png) | ![dy_pos](images/dy_pos.png) |
