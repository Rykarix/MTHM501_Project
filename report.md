# Introduction & Objectives
Are houses more expensive now than in the past and, if so, why? Is it to do with inflation? If so, to what degree? Are there other factors involved, if so, what could they be?

The objective of this report is to answer some of these questions.



# Objectives
To

# Data
The data used in this report has come from the governments price paid dataset, https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads#using-or-publishing-our-price-paid-data,

# Analysis & Results
According to various sources, https://inflation.iamkate.com/, the value of the GBP has more than doubled between 1995 & 2020. 



# Limitations
See https://www.gov.uk/guidance/about-the-price-paid-data#data-excluded-from-price-paid-data

## Excluded from the dataset by the source
There are exclusions in the data, as explained on the governments website. These exclusions are:

* sales that have not been lodged with HM Land Registry
* sales that were not for value
* transfers, conveyances, assignments or leases at a premium with nominal rent, which are:
    * ‘Right to buy’ sales at a discount
    * subject to an existing mortgage
    * to effect the sale of a share in a property, for example, a transfer between parties on divorce
    * by way of a gift
    * under a compulsory purchase order
    * under a court order
    * to Trustees appointed under Deed of appointment
* Vesting Deeds Transmissions or Assents of more than one property


## Parameters (columns) that are included in the mean
The purpose of this section is to provide transparency & demonstrate intillectual honesty in the report process. It may also highlight areas of improvement.

The scope of this report is to analyse the average price paid between counties and potentially towns. Therefore the following paremeters were not considered due to lack of relevance to the scope:
District, Locality, Street, SAON, PAON, Postcode, Transaction ID.

### Record status
This flags whether there were additions or changes to the original price. I chose to ignore this parameter for two reasons. Lack of information & lack of time due to report deadline. The main lack of information is how each 'flag' affects the price. Given more time I could analyse the difference between these flags & infer how each flag may impact factors like standard deviation & variance. This would be useful for error analysis but beyond the scope of this report.

### Duration
This parameter highlights whether a property is a 'Freehold' or 'Lease'. This was also not considered for the same reasons as above

# Conclusion
The aim of this project was to investigate housing prices since 1995 & analyze the factors that influence the average price.

# Further research
To take this report further, I'd plot the mean price onto a geospacial map of the UK, perform time analysis & attempt to create a preduction algorithm that also takes into account influencing factors such as inflation, market manipulation, supply & demand and more. Another line of enquiry would be to also analyse the cost of basic foods, toiletries & other 'essential' costs for living to analyse & compare against minimum living wage.

# References
Government PP Data: https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads

# Libraries used
Geopy - https://geopy.readthedocs.io/
