# Rwanda Environment Management Authority (REMA) RESEARCH PROJECT
# Title: Weather Direct Effects on Air Pollutants Concentration
# Written by: Jeand de Dieu Murera Gisa
# _______________________________________________

#Setting Working Directory

setwd("C:/Users/Gisa/Desktop/REMA PROJECT/AQI_data")
#Install required packages
#---------------------------------------------
# Import packages

aligning_plots_packages <- c("gridExtra", "grid")
data_exploration_packages <- c("tidyverse", "plotly", "openxlsx")
table_formating_packages <- c("knitr","kableExtra")
descriptive_statistics_packages <- c("table1","arsenal","pastecs")
air_quality_packages <- c("openair", "worldmet")
NA_treatment_packages <- c("mice", "VIM")

if (!require(install.load)) {install.packages("install.load")}

install.load::install_load(c(aligning_plots_packages,data_exploration_packages,table_formating_packages, descriptive_statistics_packages,air_quality_packages,NA_treatment_packages))
#_____________________________________________________
# Import data from openair package
# import data from the UK automatic urban and rural network in Marylebone site
#library(openair) #uncomment to run
#mary <- importAURN(site = "my1", year = 1998:2005)#uncomment to run
air_quality_dataset <- openair::mydata
head(air_quality_dataset)
#________________________________________
 #______________DATA PREPROCESSING__________
 
# Treatment of missing values
 
#Data matrixplot indicating missing values in red color

matrixplot(air_quality_dataset, sortby = 2, ylim = c(0,900), font.axis = 4)
#Number of missing values by columns

colSums(is.na(air_quality_dataset))
# Rate of missing values in data set
sum(is.na(air_quality_dataset))/(nrow(air_quality_dataset)*ncol(air_quality_dataset))
#A scatterplot with additional information on the missing values

par(
  # Change the colors
  col.main = "#336600", col.lab = "#0033FF", col.axis = "#333000",
  # Titles in italic and bold
  font.main = 4, font.lab = 4, font.axis = 4,
  # Change font size
  cex.main = 1.2, cex.lab = 1, cex.axis = 1
)
marginplot(air_quality_dataset[,c("pm10","pm25")],pch = 16 , cex = 1.5 ,numbers = T, xlim = c(0,800), ylim = c(0,400), main = "Scatterplot with missing values information", xlab = "Hourly PM10 Concentration ", ylab = "Hourly PM2.5 Concentration ")
# Rate of missing values for PM10
sum(is.na(air_quality_dataset$pm10))/(nrow(air_quality_dataset)*ncol(air_quality_dataset))
# Rate of missing values for PM2.5
sum(is.na(air_quality_dataset$pm25))/(nrow(air_quality_dataset)*ncol(air_quality_dataset))
# Deleting of NA in data set
cleaned_air_quality_dataset <- na.omit(air_quality_dataset)
head(cleaned_air_quality_dataset) # Check first 6 rows after NA deletion

# Graphical Presentation of cleaned data set
#library(Hmisc)
matrixplot(cleaned_air_quality_dataset, sortby = 2, ylim = c(0,900), font.axis = 4)

# Checking whether there is no remaining NA in data set
colSums(is.na(cleaned_air_quality_dataset))

# Viewing the descriptive statistics of PM2.5 & PM10
#we are first relabelling our columns for aesthetics.
table1::label(cleaned_air_quality_dataset$pm25) <- "Fine Particulate Matter (PM2.5)"
table1::label(cleaned_air_quality_dataset$pm10) <- "Coarse Particulate Matter (PM10)"
table1::label(cleaned_air_quality_dataset$wd) <- "Wind direction"
#Then we are creating the table with only one line of code. 
table1::table1(~pm25 + pm10 + wd, data = cleaned_air_quality_dataset)
#_________________________________________________
#VISUALIZATION OF WEATHER EFFECT ON PARTICULATE MATTER CONCENTRATION

# Wind speed Vs PM2.5
Plot1 <- ggplot(cleaned_air_quality_dataset, aes(x = ws, y = pm25)) + geom_point(col = "blue", size = 2) + theme_light() + labs(title = "Effect of wind speed on PM2.5 concentration", x = "Hourly wind speed ", y = "Hourly PM2.5 concentration", caption = "@mgisa")

# Wind speed Vs hourly PM10
Plot2 <- ggplot(cleaned_air_quality_dataset, aes(x = ws, y = pm10)) + geom_point(col = "cyan", size = 2) + theme_light() + labs(title = "Effect of wind speed on PM10 concentration", x = "Hourly wind speed ", y = "Hourly PM10 concentration",caption = "@mgisa")

# aligning two plots
grid.arrange(Plot1,Plot2, nrow = 1, top = "The wind speed effects on PM concentration", bottom = textGrob("Plotted on July 30, 2019 11: 11:30",gp = gpar(fontface = 3, fontsize = 9),hjust = 1, x = 1))
#Yearly PM2.5 concentration
polarPlot(cleaned_air_quality_dataset, pollutant = "pm25", type = "year", layout = c(4,2), key.header = "Mean of PM2.5",  cols = c("#003300","#0000FF", "#FF9933", "#FF36FF", "#FF3300"), main = " Yearly PM2.5 Concentration")

#Yearly PM10 concentration
polarPlot(cleaned_air_quality_dataset, pollutant = "pm10", type = "year", layout = c(4,2), key.header = "Mean of PM10", cols = c("#009900", "#33FFFF", "FFFF33","#FF9933", "#990033"), main = " Yearly PM10 Concentration")
# Year 2000, daily&monthly PM2.5 concentration

calendarPlot(cleaned_air_quality_dataset, pollutant = "pm25",year = "2000", cols = c("yellow","magenta", "green", "red"),key.header = "Mean of PM2.5",main = " Year 2000, daily and monthly PM2.5 distribution")

# Year 2000, daily&monthly PM2.5 concentration
calendarPlot(cleaned_air_quality_dataset, pollutant = "pm10", year = "2000", cols = c("yellow","cyan", "blue", "red"),key.header = "Mean of PM10",main = " Year 2000, daily and monthly PM10 distribution")
# Seasonal PM2.5 concentration
pollutionRose(cleaned_air_quality_dataset, pollutant = "pm25", key.header = "PM2.5 concentration",cols = c("yellow", "green", "blue", "black", "red"), type = 'season', legend_title = "PM2.5 concentration",legend.title.align = .5, angle = 45, width = 1,grid.line = list(value = 10, lty = 5, col = "purple"),main = "Seasonal concentration of PM2.5")

# Seasonal PM10 concentration
pollutionRose(cleaned_air_quality_dataset, pollutant = "pm10", key.header = "PM10 concentration",cols = c("yellow", "green", "blue", "black", "red"), type = 'season', legend_title = "PM10 concentration",legend.title.align = .5, angle = 45, width = 1,grid.line = list(value = 10, lty = 5, col = "red"),main = "Seasonal concentration of PM10")
# PM2.5 in weekdays and weekend
polarPlot(cleaned_air_quality_dataset, pollutant = "pm25", type = "weekend",cols = c("yellow", "blue", "magenta"),key.header = "Mean of PM25",main = "PM25 distribution in weekdays and weekend")

# PM10 in weekdays and weekend
polarPlot(cleaned_air_quality_dataset, pollutant = "pm10", type = "weekend",cols = c("yellow", "blue", "red"),key.header = "Mean of PM10",main = "PM10 distribution in weekdays and weekend")
# Daily PM2.5 concentration
polarPlot(cleaned_air_quality_dataset, pollutant = "pm25", type = "weekday",cols = c("yellow", "blue", "magenta"),key.header = "Mean of PM2.5",main = " Daily PM2.5 distribution")

# Daily PM10 concentration
polarPlot(cleaned_air_quality_dataset, pollutant = "pm10", type = "weekday",cols = c("yellow", "blue", "red"),key.header = "Mean of PM10",main = " Daily PM10 distribution")

#Day&nighttime PM2.5 concentration
polarPlot(cleaned_air_quality_dataset, pollutant = "pm25", type = "daylight",cols = c("yellow", "green", "magenta"),key.header = "Mean of PM2.5",main = " Day and night PM2.5 distribution")


# Day&nighttime PM10 concentration
polarPlot(cleaned_air_quality_dataset, pollutant = "pm25", type = "daylight",cols = c("yellow", "blue", "red"),key.header = "Mean of PM10",main = " Day and night PM10 distribution")
#_______________________________________________________

#CORRELATION TECHNIQUES TO IDENTIFY THE PM MAIN SOURCES

# Pearson Correlation btn PM2.5 and PM10

polarPlot(cleaned_air_quality_dataset, pollutant = c("pm25","pm10"), statistic = "r",cols = c("yellow", "blue", "green", "red"), key.header = "Correlation coefficient", main = " Pearson correlation between PM2.5 and PM10")
