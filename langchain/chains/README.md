## Chains
Chains are used to build pipelines for LLM applications, where the output of one component becomes the input of the next component. This simplifies the code, reduces development complexity, and makes applications more efficient and easier to maintain.

```
chain = prompt | model | output_parser
```

There are three main types of chains:

1. **Sequential Chain (Simple):** Components are executed one after another, where each step uses the output of the previous step.
```
      +-------------+      
      | PromptInput |      
      +-------------+      
             *             
             *             
             *             
    +----------------+     
    | PromptTemplate |     
    +----------------+     
             *             
             *             
             *             
+------------------------+ 
| ChatGoogleGenerativeAI | 
+------------------------+ 
             *             
             *             
             *             
    +-----------------+    
    | StrOutputParser |    
    +-----------------+    
             *             
             *             
             *             
+-----------------------+  
| StrOutputParserOutput |  
+-----------------------+  
             *             
             *             
             *             
    +----------------+     
    | PromptTemplate |     
    +----------------+     
             *             
             *             
             *             
+------------------------+ 
| ChatGoogleGenerativeAI | 
+------------------------+ 
             *             
             *             
             *             
    +-----------------+    
    | StrOutputParser |    
    +-----------------+    
             *             
             *             
             *             
+-----------------------+  
| StrOutputParserOutput |  
+-----------------------+ 
```
2. **Parallel Chain:** Multiple components run simultaneously, and their outputs can be combined later.
```
                +--------------------------------+               
                | Parallel<notes,questions>Input |               
                +--------------------------------+               
                      ***                 ****                   
                  ****                        ***                
                **                               **              
    +----------------+                      +----------------+   
    | PromptTemplate |                      | PromptTemplate |   
    +----------------+                      +----------------+   
             *                                        *          
             *                                        *          
             *                                        *          
+------------------------+                  +-----------------+  
| ChatGoogleGenerativeAI |                  | ChatHuggingFace |  
+------------------------+                  +-----------------+  
             *                                        *          
             *                                        *          
             *                                        *          
    +-----------------+                     +-----------------+  
    | StrOutputParser |                     | StrOutputParser |  
    +-----------------+                     +-----------------+  
                      ***                 ****                   
                         ****          ***                       
                             **      **                          
                +---------------------------------+              
                | Parallel<notes,questions>Output |              
                +---------------------------------+              
                                 *                               
                                 *                               
                                 *                               
                        +----------------+                       
                        | PromptTemplate |                       
                        +----------------+                       
                                 *                               
                                 *                               
                                 *                               
                    +------------------------+                   
                    | ChatGoogleGenerativeAI |                   
                    +------------------------+                   
                                 *                               
                                 *                               
                                 *                               
                        +-----------------+                      
                        | StrOutputParser |                      
                        +-----------------+                      
                                 *                               
                                 *                               
                                 *                               
                     +-----------------------+                   
                     | StrOutputParserOutput |                   
                     +-----------------------+ 
```
3. **Conditional Chain:** The next component is selected based on a condition or the result of a previous step.
```
    +-------------+      
    | PromptInput |      
    +-------------+      
            *            
            *            
            *            
   +----------------+    
   | PromptTemplate |    
   +----------------+    
            *            
            *            
            *            
  +-----------------+    
  | ChatHuggingFace |    
  +-----------------+    
            *            
            *            
            *            
+----------------------+ 
| PydanticOutputParser | 
+----------------------+ 
            *            
            *            
            *            
       +--------+        
       | Branch |        
       +--------+        
            *            
            *            
            *            
    +--------------+     
    | BranchOutput |     
    +--------------+ 
```