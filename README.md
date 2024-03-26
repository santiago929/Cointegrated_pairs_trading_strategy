# Disclaimer.

This repository contains code that may serve as a reference or example for completing assignments. While the code provided here may offer potential solutions, it is important to acknowledge that it is intended for educational and reference purposes only.

- Educational Reference: This code is meant to provide guidance and understanding of certain concepts and techniques. It should not be used verbatim as a solution for assignments without understanding and modification.

- Personal Use: You are welcome to study, modify, and utilize this code for personal learning or academic purposes. However, it is your responsibility to ensure compliance with any academic integrity policies or guidelines set forth by your institution.

- No Guarantee: While efforts have been made to ensure the accuracy and reliability of the code provided, there is no guarantee that it will be error-free or suitable for your specific requirements. You are encouraged to test and validate any code before using it in a production environment or for critical tasks.

# Contents.

We make use of an already-made code exposed in Halls-Moore (2015) for data handling and portfolio tracking, although some changes were made to adapt the trading rules to our strategy. 

- Code ONE file contains the calculations of the Engle-Granger process, cointegrated residual, error correction equation and finally, it assesses the quality of reversion.
- Code TWO is where we take the results from code ONE and adapt them into the strategy.
  
  - Note that the parameters 𝛽′𝐶𝑜𝑖𝑛𝑡, 𝜇_𝑒 𝑎𝑛𝑑 𝜎_𝑒𝑞 will remain fixed as the optimization process is just made for Z. 
  - Code file TWO is amended to handle position sizing according to the Engle-Granger process, whilst maintaining the author’s proposed trading orders irrespective of cash held, contract and tick size (most common for futures trading) and margin requirements;
    as well, we do not consider any commissions charged by the broker, the depth of the order book nor the bid/ask spreads as the liquidity of some commodities futures like Copper and Platinum contracts is rather low. Therefore, the fill cost is set to the current market      price which is the closing price of the day before, and the number of contracts (set to 100 as default).

# References.

Diamond, R. V. (2014). Learning and Trusting Cointegration in Statistical Arbitrage. Social 
Science Research Network, 27. doi:https://dx.doi.org/10.2139/ssrn.2220092

Halls-Moore , M. L. (2015). Successful Algorithmic Trading. Retrieved 07 2023, from 
https://www.quantstart.com/successful-algorithmic-trading-ebook/

# Logic behind the strategy. 


Therefore, we start with the following time series equation:

█(y_t= ϵ_t+ βy_(t-1)  #(1) )

Decomposing the above equation renders:

y_t= ϵ_t+ β(ϵ_(t-1)+ βy_(t-2))

y_t= ϵ_t+βϵ_(t-1)+ β^2 (ϵ_(t-2)+ βy_(t-3))

█(y_t= ϵ_t+βϵ_(t-1)+ β^2 ϵ_(t-2)+⋯#(1.1) )


By following the decomposition method, we are left with just the residuals as shown in (1.1). Furthermore, we can note that the impact of past residuals fades as i ~ ∞:

█(y_t= ∑_(i=0)^p▒β^i  ϵ_(t-j)  #(1.1.1) )
The null hypothesis is that time series has a unit root. Therefore:

H_0: Y_t= Y_(t-1)+ ϵ_t     implies   β=1,∆Y_t= ϵ_t

H_1: Y_t= βY_(t-1)+ ϵ_t     substract   Y_(t-1)

We test for significance of ø= β-1 by comparing the t-statistic to a critical value taken from the Dickey-Fuller distribution. Hence: 

∆Y_t= øY_(t-1)+ ϵ_t   

Failing to reject the Null Hypothesis (H0) confirms that the time series has a unit root given:

ø=1-β=0  →   β=1  ∴  ∆Y_t= ϵ_t


