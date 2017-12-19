# Clean environment and load required packages
rm(list=ls())
cat("\014")
require(EBImage)
library(data.table)

#Read training file
setwd("~/darknet-master/build/darknet/x64/results/")
file_name = "keeper_20_last_chanceconfidence_v4.csv"
data <- read.csv(file_name,header=TRUE,na.strings="",row.names=NULL)   

# Main loop. Loop over each image
setwd("~/drivendata/fish/cropped_test_images/")

# Set width
w <- 32
# Set height
h <- 32

# Set up df
df <- data.frame()
my.list <- vector(mode="list")
colnames(data) <- c("filenames","score","x1","y1","x2","y2")
for(i in 1:dim(data)[1])
{
  filename = paste(data$filename[i],".jpg",sep="")
  #print(filename)
  # Read image
  img <- readImage(filename)
  #img <- normalize(img)
  img <- channel(img,"gray")
  img <- resize(img,w,h)
  # Get the image as a matrix
  img_matrix <- matrix(img@.Data)
  # Coerce to a vector
  img_vector <- as.vector(t(img_matrix))
  #vec <- c(img_vector)
  # Bind rows
  my.list[[i]]<-img_vector
  #df <- rbind(df,img_vector)
  # Print status info
  if (i == 34369)
  {
    print ("25% complete")
  }
  if (i == 68738)
  {
    print ("50% complete")
  }
  if (i == 103122)
  {
    print ("75% complete")
  }
  #print(paste("Done ", i, sep = ""))
}
# Set image size. 
print("Complete dataframe creation")
df <- rbind(df, do.call(rbind, my.list))

img_size <- w*h

# Set names
names(df) <- paste("pixel", c(1:img_size))

# Out file

out_file <- paste("~/drivendata/fish/","crop_test_20_last_chance","_",w,"x",h,".csv",sep="")

# Write out dataset
write.csv(df, out_file, row.names = FALSE)



