#
# This is a Shiny web application. You can run the application by clicking
# the 'Run App' button above.
#
# Find out more about building applications with Shiny here:
#
#    http://shiny.rstudio.com/
#
library(shiny)
library(DT)
library(data.table)
library(torch)
# library(ggplot2)
#library(datasets)
# library(reshape)
# library(shinycssloaders)

library(magrittr)
# library(markdown)
# library(bslib)
library(dplyr)
# library(ggpubr)
# library(ggrepel)
library(shinythemes)
#library(data.table)
# library(magrittr)
# library(reticulate)

options(bitmapType='cairo')
##### Initals / startup code #################################################
main.text2 = "The website provides interface for user to our computational approach where users can predict patient response to immunotherapy in arbitrary melanoma cohorts using gene expression from patients as input." 

qnorm.array <- function(mat)
{
  mat.back = mat
  mat = mat.back[!is.na(mat.back)]
  mat = rank(mat,  rank, ties.method = "average");
  mat = qnorm(mat / (length(mat)+1));
  mat.back[!is.na(mat.back)] = mat
  mat.back
}
process.input <- function(mat) {
  mat = apply(mat,2,qnorm.array) 
  mat[is.na(mat)] = 0
  mat
}

# immune_metabolism =TRUE

# install_torch()
immune_metabolism_ensembl_input_genes = c('ENSG00000072364',
                                          'ENSG00000189079',
                                          'ENSG00000204256',
                                          'ENSG00000134058',
                                          'ENSG00000118260',
                                          'ENSG00000101412',
                                          'ENSG00000169016',
                                          'ENSG00000120690',
                                          'ENSG00000151702',
                                          'ENSG00000101216',
                                          'ENSG00000125651',
                                          'ENSG00000164683',
                                          'ENSG00000117139',
                                          'ENSG00000103495',
                                          'ENSG00000025434',
                                          'ENSG00000185551',
                                          'ENSG00000123358',
                                          'ENSG00000140464',
                                          'ENSG00000117222',
                                          # 'avi',
                                          'ENSG00000111424')

angiogenesis_ensembl_input_genes = c("ENSG00000156127", "ENSG00000082258", "ENSG00000164330", "ENSG00000160973", "ENSG00000174332", "ENSG00000171988", "ENSG00000103495", "ENSG00000197157", "ENSG00000269404", "ENSG00000073861", "ENSG00000187079", "ENSG00000198176", "ENSG00000073282", "ENSG00000125482")


# Define UI for application that draws a histogram
ui <- fluidPage(
  theme = shinytheme("journal"),
  tags$head(HTML('<link rel="icon", href="icon.png", type="image/png" />')),
  
  navbarPage(
    title = div( "BTAS", style='color:black;font-size:150%'),
    p(HTML(main.text2)),

    br(),
    
    sidebarLayout(
      sidebarPanel(
        fileInput("file1", "Please input your gene expression file", accept = ".txt"),
        checkboxInput("first_col_name", "First column patient name in input?", FALSE),
        
        selectInput("immune_metabolism", h3("Select type of bipotent targets"), 
                    choices = list("Immune-metabolism" = TRUE, "Angiogenesis/Growth-suppressor" = FALSE), selected = TRUE)
        
        # ,
        # Button
        # downloadButton("downloadData", "Download")
      ),
      mainPanel(

        
      
      # div(img(src="icon.png", height =200, width = 200),(HTML(main.text2))),
      # p(HTML(main.text2)),
      # downloadLink("downloadData", "Download"),
        fluidRow(
          column(6,
                 # div(img(src="icon.png", height =200, width = 200))
                 img(src="icon.png", height="60%", width="60%", align="right")
          ),
          column(6,
                 downloadButton("downloadData", "Download predicted scores")
          )
        ),
     
      span(textOutput("textout"), style="color:red;font-size:120%"),
  
      tableOutput("table")
    )
  )
)
)



server <- function(input, output) {
  
  run.model <- reactive({
    file <- input$file1
    ext <- tools::file_ext(file$datapath)
    
    
    # req(file)
    validate(need(ext == "txt", ""))
    
    # read.csv(file$datapath, header = input$header)
    
    inp = fread(file$datapath)
    ## store patient name 
    patient.name = paste0("Pt", seq(nrow(inp))) 
    if(input$first_col_name){
      patient.name = inp[,1,with=F] %>% unlist
      
    }
    
    if(input$immune_metabolism){
      input.genes = immune_metabolism_ensembl_input_genes 
      saved_model_dir = "./pretrainted_models/immune_metabolism/"
      header = "Risk based on Immune-metabolism deepBTAS"
    }else{
      input.genes = angiogenesis_ensembl_input_genes
      saved_model_dir = "./pretrainted_models/angiogenesis/"
      header = "Risk based on Angiogenesis/Growth-suppressor deepBTAS"
    }
    
    
    embedding_path = sprintf("%s/embedding_model_scripted.pt" , saved_model_dir) 
    output_model_path = sprintf("%s/outputs_scripted.pt" , saved_model_dir)
    # embedding_model =  torch::jit_load(embedding_path)
    # outputs = torch::jit_load(output_model_path)
    # torch::torch_save(embedding_model, "embedding_model.pt")
    # torch::torch_save(outputs, "outputs.pt")
# browser()
    # load("./model.RData")
    embedding_model =  torch::torch_load("embedding_model.pt")
    outputs =  torch::torch_load("outputs.pt")
    embedding_model$eval()
    outputs$eval()
    inp.col = colnames(inp)
    diff.genes = setdiff(input.genes, inp.col)
    textout = ""
    if(length(diff.genes) > 0) {
      inp.temp = matrix(NA, ncol=length(diff.genes), nrow=nrow(inp))
      diff.genes.str = paste(diff.genes, collapse=",")
      textout= sprintf("Warning! Unreliable output because of genes %s missing from the input. Setting missing genes to 0. ", diff.genes.str)
      colnames(inp.temp) = diff.genes
      inp =cbind(inp, inp.temp)
    }
    inp.mat = inp[,input.genes,with=F] %>% as.matrix %>% process.input
    data_batch = torch_tensor(inp.mat,dtype=torch_float32())
    
    embedding_batch = embedding_model(data_batch)
    output_batch = outputs(embedding_batch)
    
    out  = as_array(output_batch) %>% as.data.table %>% 
      set_colnames(header)
    out$patient= patient.name
    out= out[,2:1, with=F]
    list(out=out, textout=textout)
  })
  
  # Table of selected dataset ----
  output$table <- renderTable({
    run.model() %>% .[["out"]]
    
  })
  
  output$textout <- renderText({ 
    run.model() %>% .[["textout"]]
  })
  
  # Downloadable csv of selected dataset ----
  output$downloadData <- downloadHandler(
    filename = function() {
      paste(input$dataset, ".csv", sep = "")
    },
    content = function(file) {
      write.csv(run.model()%>% .[["out"]], file, row.names = FALSE)
    }
  )
}

shinyApp(ui, server)