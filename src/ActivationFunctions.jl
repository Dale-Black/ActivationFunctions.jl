module ActivationFunctions

export binary, sigmoid, scaled_sigmoid, probit, vsf, sss, tanh36, tanh3, scaled_tanh, 
    lrtanh, nsigmoid, improved_logsig, siglin, ptanh, srs, arctan, arctangr, sigalg
    # tssig, rsigelu, hardsrelue, elish, hardelish, rsigelud, lsrelu, sqnl, sqlu, squish,
    # sqreu, sqsoftplus, logsqnl, linq, isrlu, isru, mef, sqrt_af, ssaf, bent, mishra,
    # sbaf, laf, symexp, spocu, puaf, softplus, psoftplus, softpp, rsp, arandaordaz,
    # bfire, bbfire, pmaf, prbf, combhsine, marcsinh, hypersinh, arctid, sine, shifted_sine,
    # cosine, cosid, sinp, gcu, asu, sinc, ssu, dsu, hclsh, polyexp, exponential, etanh,
    # wave, ncu, triple, squ, kdac, kwta, vbaf, hcaf, fcaf, ccaf, sqnl_smooth, sqlu_smooth,
    # squish_smooth, sqreu_smooth, sqsoftplus_smooth, logsqnl_smooth, linq_smooth, 
    # isrlu_smooth, isru_smooth, mef_smooth, sqrt_af_smooth, ssaf_smooth, bent_smooth, 
    # mishra_smooth, sbaf_smooth, laf_smooth, symexp_smooth, spocu_smooth, puaf_smooth,
    # softplus_smooth, psoftplus_smooth, softpp_smooth, rsp_smooth, arandaordaz_smooth, 
    # bfire_smooth, bbfire_smooth, pmaf_smooth, prbf_smooth, combhsine_smooth,  
    # marcsinh_smooth, hypersinh_smooth, arctid_smooth, sine_smooth, shifted_sine_smooth,
    # cosine_smooth, cosid_smooth, sinp_smooth, gcu_smooth, asu_smooth, sinc_smooth,
    # ssu_smooth, dsu_smooth, hclsh_smooth, polyexp_smooth, exponential_smooth,  
    # etanh_smooth, wave_smooth, ncu_smooth, triple_smooth, squ_smooth, kdac_smooth,
    # kwta_smooth, vbaf_smooth, hcaf_smooth, fcaf_smooth, ccaf_smooth, relu, shifted_relu,
    # lrelu, vlrelu, rrelu, srrelu, slrelu, nrelu, sinerelu, minsin, vlu, scaa, rtrelu,
    # appelu, stepplus, lreluplus, vreluplus, srelu, birelu, hardsigmoid, hardtanh,  
    # shiftedhtanh, hardswish, trec, hardshrink, softshrink, blrelu, vrelu, pan, abslu,
    # mrelu, lsptlu, lrtlu, softmodulusq, softmodulust, signrelu, lirelu, crelu, ncrelu,
    # dualrelu, oplu, erelu, repu, apprelu, plaf, eplaf, oplaf, abrelu, drlu, disrelu, 
    # mlrelu, fts, reluswish, oaf, elu, reu, ada, lada, nlrelu, slu, resp, prenu, pelu,
    # pelu_mix, relu_pp, relu_psoftplus, relu_pelu2, relu_mpelu, p_e2_relu, relu_p2id,
    # relu_p2relu1, relu_psrelu, relu_plsrelu, delu, shelue, svelue, pshelue, psvelue,  
    # sc_mish, scl_mish, sc_swish, pswish, pfts, pfpm, rsigelu, hardsrelue, elish, 
    # hardelish, rsigelud, ls_relu, selu, lselu, serlu, sselu, sSELU, rsigelu, rsigelu_d,
    # celu, erfrelu, pselu, lpselu, lpselurp, t_swish, relu_taaf, ptanhexp, siglu, sara,
    # maxsig, thlu, dualelu, diffelu, polylu, fplus, relu_p2elu, iplus, lisht, mish, smish,
    # tanhexp, eacu, gelu, sgelu, calu, lalu, pgelu, geu, pelu, relu_pswish, eswish,  
    # aconb, aconc, acon_meta, psgu, relu_ptsrelu, relu_ptbsrelu, pats, aqulu, sinlu, 
    # erfact, pserf, swim, pwlu, npwlu, cpn, cpn_nl, cpn_mc, mtlu, sau, smu, lelelu, preu, 
    # rtprelu, dprelu, funrelu, funprelu, rprelu, probact, aoaf, dlrelu, drelu, frelu, 
    # shilu, starrelu, htanh, arelu, relu_dpl, fpaf, relu_fpaf, eprelu, relu_paired, 
    # relu_tent, relu_hat, rmaf, ptelu, relu_ptelu, talu, ptalu, relu_ptanhlu, reltanh,
    # blu, reblu, relu_aplu, ada_spline, adaptivefunc, naf, naf_mix, zwnaf, relu_adaptive,
    # relu_adaptive_poly, vaf, relu_vaf, relu_flex_dyna, relu_naf, relu_dyna, dknn,
    # relu_dknn_rowdy, slaf, relu_ps, chpaf, lpaf, hpaf, msrff, tru, relu_msmo, softsign,
    # scaled_softsign, ssrlu, splusroottwol, s_sig, s_tanh, s_arctan, s_bentid, s_isru,
    # s_isrlu, s_laf, s_softplus, s_elu, s_selu, s_gelu, s_li, s_mila, s_relu_swish,
    # s_squareplus, s_relu_pswish, s_stepplus, s_lreluplus, s_elu, s_vreluplus,  
    # s_sreluplus, s_alisa, s_allrelu, s_plu, s_dyna_plu, s_adalu, s_tsaf, s_melu, 
    # s_mmelu, s_galu, s_sc_mish, s_relu_scli_mish, s_sc_swish, s_pswish, s_pfts, s_pfpm, 
    # maxout, abu, abu_mix, abu_gate, abu_ndc, abu_dknn, glu, gtu, reglu, geglu, swiglu,
    # softmax, tuned_sm, glsoftmax, gpsoftmax, arbf, pgelu

"""
## `binary`

```julia  
binary(z) = z >= 0 ? 1 : 0
```
    
Binary activation function.

#### Arguments
- `z`: Input value.

#### Returns 
- 1 if `z` ≥ 0, 0 otherwise.
"""
binary(z) = z >= 0 ? 1 : 0

"""
## `sigmoid`

```julia
sigmoid(z) = 1 / (1 + exp(-z))  
```

Logistic sigmoid activation function.

#### Arguments  
- `z`: Input value.

#### Returns
- Value between 0 and 1, with sigmoid(0) = 0.5.  
"""
sigmoid(z) = 1 / (1 + exp(-z))

"""
## `scaled_sigmoid`

```julia
scaled_sigmoid(z) = 4*sigmoid(z) - 2 
```

Scaled sigmoid activation function.

#### Arguments
- `z`: Input value. 

#### Returns
- Value between -2 and 2.
"""
scaled_sigmoid(z) = 4*sigmoid(z) - 2

"""
## `probit`

```julia
probit(z) = cdf(Normal(0,1), z)  
```

Probit activation function. Equivalent to cumulative distribution function of standard normal distribution.

#### Arguments  
- `z`: Input value.

#### Returns  
- Value between 0 and 1.
"""
probit(z) = cdf(Normal(0,1), z)

"""
## `vsf`

```julia
vsf(z; a=1, b=1, c=0) = a*sigmoid(b*z) - c
```

Variant sigmoid function (VSF).  

#### Arguments
- `z`: Input value. 
- `a`: Scaling parameter. Default is 1.
- `b`: Slope parameter. Default is 1.
- `c`: Shift parameter. Default is 0.

#### Returns
- Sigmoid-like output value.  
"""
vsf(z; a=1, b=1, c=0) = a*sigmoid(b*z) - c

"""
## `sss`

```julia 
sss(z; a=1, b=0) = sigmoid(a*(z-b))
```  

Shifted and scaled sigmoid (SSS) activation function.

#### Arguments
- `z`: Input value.
- `a`: Slope parameter. Default is 1. 
- `b`: Shift parameter. Default is 0.

#### Returns 
- Sigmoid output shifted by `b` and scaled by `a`.
"""
sss(z; a=1, b=0) = sigmoid(a*(z-b))

"""
## `tanh36`

```julia
tanh36(z) = 0.5tanh(2z)+0.5  
```

Efficient tanh approximation using 36 equidistant points.

#### Arguments
- `z`: Input value.

#### Returns
- Approximate tanh value between 0 and 1.
"""
tanh36(z) = 0.5tanh(2z)+0.5

"""  
## `tanh3`

```julia
tanh3(z) = 0.5tanh(0.5z)+0.5
```

Efficient tanh approximation using 3 points.

#### Arguments
- `z`: Input value.

#### Returns
- Approximate tanh value between 0 and 1.
"""
tanh3(z) = 0.5tanh(0.5z)+0.5

"""
## `scaled_tanh`

```julia
scaled_tanh(z; a=1.7159, b=2/3) = a*tanh(b*z)
```

Scaled tanh activation function.

#### Arguments
- `z`: Input value.
- `a`: Output scaling parameter. Default is 1.7159.
- `b`: Input scaling parameter. Default is 2/3.  

#### Returns
- Scaled tanh output.
"""
scaled_tanh(z; a=1.7159, b=2/3) = a*tanh(b*z)

"""
## `lrtanh`

```julia
lrtanh(z) = tanh(z/2)
```

LRTanh activation function. Substitute tanh derivative in backpropagation.

#### Arguments
- `z`: Input value.

#### Returns  
- tanh(z/2)
"""
lrtanh(z) = tanh(z/2)

"""
## `nsigmoid`

```julia
nsigmoid(z; a=0.02, b=600) = sigmoid(a*(z-b)) + sigmoid(-a*(z+b)) - 1  
```

n-sigmoid activation function.

#### Arguments
- `z`: Input value.
- `a`: Slope parameter. Default is 0.02.
- `b`: Shift parameter. Default is 600.

#### Returns
- Sum of two shifted sigmoids minus 1.
"""
nsigmoid(z; a=0.02, b=600) = sigmoid(a*(z-b)) + sigmoid(-a*(z+b)) - 1

"""
## `improved_logsig`

```julia
function improved_logsig(z; a=0.1, b=1)
    if z >= b
        return a*(z-b) + sigmoid(b)
    elseif -b < z < b
        return sigmoid(z) 
    else
        return a*(z+b) - sigmoid(b)
    end
end
```

Improved logistic sigmoid activation function.

#### Arguments
- `z`: Input value.
- `a`: Slope parameter. Default is 0.1.  
- `b`: Threshold parameter. Default is 1.

#### Returns
- Piecewise combination of linear functions and sigmoid.
"""
function improved_logsig(z; a=0.1, b=1)
    if z >= b
        return a*(z-b) + sigmoid(b)
    elseif -b < z < b
        return sigmoid(z) 
    else
        return a*(z+b) - sigmoid(b)
    end
end

"""
## `siglin`

```julia
siglin(z; a=0.05) = sigmoid(z) + a*z
```

Combination of sigmoid and linear functions.  

#### Arguments
- `z`: Input value.
- `a`: Linear scaling parameter. Default is 0.05.

#### Returns
- Sum of sigmoid and linear.
""" 
siglin(z; a=0.05) = sigmoid(z) + a*z

"""
## `ptanh`

```julia
ptanh(z; a=0.01, b=0.01) = tanh(az) + bz*sech(z)^2
```

Parametric tanh function.

#### Arguments  
- `z`: Input value.
- `a`: tanh scaling parameter. Default is 0.01.
- `b`: Linear scaling parameter. Default is 0.01.

#### Returns
- Parametric tanh output.
"""
ptanh(z; a=0.01, b=0.01) = tanh(az) + bz*sech(z)^2

"""
## `srs`

```julia  
srs(z; a=2, b=3) = z/(z^a + exp(-z/b))
```

Soft-root-sign (SRS) activation function.

#### Arguments
- `z`: Input value.
- `a`: Power parameter. Default is 2.
- `b`: Exponential scaling parameter. Default is 3.

#### Returns
- SRS output.
"""
srs(z; a=2, b=3) = z/(z^a + exp(-z/b))  

"""
## `arctan`

```julia
arctan(z) = atan(z)
```

Arctan activation function.

#### Arguments
- `z`: Input value.

#### Returns
- Arctan of input, in radians.  
"""
arctan(z) = atan(z)

"""
## `arctangr`

```julia
arctangr(z) = atan(z) / (1+sqrt(2)/2)
```

Arctangr activation function.

#### Arguments
- `z`: Input value.

#### Returns
- Scaled arctan of input.
"""
arctangr(z) = atan(z) / (1+sqrt(2)/2)

"""
## `sigalg`

```julia
sigalg(z; a=0.5) = 1 / (1 + exp(-(z*(1+a*abs(z))) / (1+abs(z)*(1+a*abs(z)))))
```

Sigmoid-Algebraic activation function.  

#### Arguments
- `z`: Input value.  
- `a`: Scaling parameter. Default is 0.5.

####
"""
sigalg(z; a=0.5) = 1 / (1 + exp(-(z*(1+a*abs(z))) / (1+abs(z)*(1+a*abs(z)))))


end # module