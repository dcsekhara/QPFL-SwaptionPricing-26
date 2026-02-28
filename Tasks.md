Tasks: 
- Future values(main)
- Impute missing values(optional) 

Dataset: 
- Synthetic yet generated realistically.
- Underlying is interest rate swaps. 

Params: 
- Maturity - Maturity of the underlying option, 
  - These are european options. 
- Tenor - How long the payments will be made. From the date of maturity maybe??

Expected Techniques: 
- Quantum resorvoir or similar QML techniques. 


- Strike date of option is the expiration date
- Swap rate is the decided rate of interest to be swapped
- Market rate is the LIBOR - floating rate
- Option period is until the strike date


- Assuming the strike rate is same for all the swaptions listed.
- I think its best to go prediction via yield curve.
