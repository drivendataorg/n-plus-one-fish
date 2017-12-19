# Clean environment and load required packages
rm(list=ls())
cat("\014")
require(EBImage)

#Read training file
setwd("~/drivendata/fish/")
file_name = "crop_training.csv"
data <- read.csv(file_name,header=TRUE,na.strings="",row.names=NULL)   

label <- vector()
for (n in 1:dim(data)[1])
{
  if (data$species_fourspot[n] == 1)
  {
    label[n] = 1
  }
  else if (data$species_greysole[n] == 1)
  {
    label[n] = 2
  }
  else if (data$species_other[n] == 1)
  {
    label[n] = 3
  }
  else if (data$species_plaice[n] == 1)
  {
    label[n] = 4
  }
  else if (data$species_summer[n] == 1)
  {
    label[n] = 5
  }
  else if (data$species_windowpane[n] == 1)
  {
    label[n] = 6
  }
  else if (data$species_winter[n] == 1)
  {
    label[n] = 7
  }
}

# Main loop. Loop over each image
setwd("~/drivendata/fish/cropped_training_images/")

# Set width
w <- 32
# Set height
h <- 32

# Set up df
df <- data.frame()
my.list <- vector(mode="list")
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
  if (i == 12022)
  {
    print ("25% complete")
  }
  if (i == 24044)
  {
    print ("50% complete")
  }
  if (i == 30056)
  {
    print ("75% complete")
  }
  #print(paste("Done ", i, sep = ""))
}
# Set image size. 
print("Complete dataframe creation")
df <- rbind(df, do.call(rbind, my.list))
df <- cbind(label,df)

img_size <- w*h

# Set names
names(df) <- c("label",paste("pixel", c(1:img_size)))

# Out file

out_file <- paste("~/drivendata/fish/","crop_data_v3","_",w,"x",h,".csv",sep="")

# Write out dataset
write.csv(df, out_file, row.names = FALSE)



