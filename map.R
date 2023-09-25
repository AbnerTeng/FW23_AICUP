library(tmap)
library(sf)

# Read data and set as sf format
house <- read.csv("./data/training_data.csv") %>% 
  st_as_sf(coords = c("橫坐標", "縱坐標"), 
           crs = 3826)

read_external_data <- function(file_name){
  paste("./data/external_data/", file_name, sep = "") %>% 
    read.csv(.) %>% 
    st_as_sf(coords = c("lng", "lat"), 
             crs = 4326, 
             na.fail = F) %>% 
    st_transform(crs = 3826) %>% 
    return()
}

bus <- read_external_data("公車站點資料.csv")
train <- read_external_data("火車站點資料.csv")
bike <- read_external_data("腳踏車站點資料.csv")
MRT <- read_external_data("捷運站點資料.csv")

ATM <- read_external_data("ATM資料.csv")
conv_store <- read_external_data("便利商店.csv")

ele_school <- read_external_data("國小基本資料.csv")
junior_school <- read_external_data("國中基本資料.csv")
high_school <- read_external_data("高中基本資料.csv")
college <- read_external_data("大學基本資料.csv")

post_office <- read_external_data("郵局據點資料.csv")
hospital <- read_external_data("醫療機構基本資料.csv")
bank <- read_external_data("金融機構基本資料.csv")

# Make interactive map
tmap_mode("view")
tm_shape(house) + 
  tm_dots(col = "單價") +
  # tm_shape(bus) + 
  # tm_dots(col = "#4E79A7", id = "站點UID") +
  tm_shape(train) + 
  tm_dots(col = "#A0CBE8", id = "站點名稱") +
  tm_shape(bike) + 
  tm_dots(col = "#F28E2B", id = "站點名稱") +
  tm_shape(MRT) + 
  tm_dots(col = "#FFBE7D", id = "站點UID") +
  tm_shape(ATM) + 
  tm_dots(col = "#59A14F", id = "裝設金融機構名稱") +
  tm_shape(conv_store) + 
  tm_dots(col = "#8CD17D", id = "公司名稱") +
  tm_shape(ele_school) + 
  tm_dots(col = "#B6992D", id = "學校名稱") +
  tm_shape(junior_school) + 
  tm_dots(col = "#F1CE63", id = "學校名稱") +
  tm_shape(high_school) + 
  tm_dots(col = "#499894", id = "學校名稱") +
  tm_shape(college) + 
  tm_dots(col = "#86BCB6", id = "學校名稱") +
  tm_shape(post_office) + 
  tm_dots(col = "#E15759", id = "局名") +
  tm_shape(hospital) + 
  tm_dots(col = "#FF9D9A", id = "機構名稱") +
  tm_shape(bank) + 
  tm_dots(col = "#B07AA1", id = "金融機構名稱") +
  tm_add_legend(
    type = "fill",
    labels = c(
      #"公車站", 
      "火車站", "腳踏車站", "捷運站",
      "ATM", "便利商店",
      "國小", "國中", "高中", "大學",
      "郵局", "醫療機構", "金融機構"
      ),
    col = c(
      #"#4E79A7", 
      "#A0CBE8", "#F28E2B", "#FFBE7D", 
      "#59A14F", "#8CD17D",
      "#B6992D", "#F1CE63", "#499894", "#86BCB6",
      "#E15759", "#FF9D9A", "#B07AA1")
    )
