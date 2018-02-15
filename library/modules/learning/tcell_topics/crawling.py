'''
Created on Feb 14, 2017

@author: dicle
'''

from bs4 import BeautifulSoup
import requests, re

def get_link_labels(html):
    soup = BeautifulSoup(html)
    labels = [a.text for a in soup.findAll("a")]
    return labels



def collect_all_links():
    
    main_url = "http://www.turkcell.com.tr/yardim"
    pages = collect_links1(main_url)
    pages2 = []
     
    for page in pages:
        sublinks = page["sublinks"]
        main_title = page["title"]
        #print(main_title)
        page2 = page.copy()
        
        subpages = []
        for title, link in sublinks:
            
            subpage = {}
            subpage["title"] = title
            subpage["link"] = link
            
            
            links = collect_sublinks(link)
            
            subpage["sublinks"] = links
            subpages.append(subpage)
        
        page2["sublinks"] = subpages
        pages2.append(page2)
    
    return pages2

def collect_links1(url):
    
    head = "http://www.turkcell.com.tr/"
    
    pages = []
    
    req = requests.get(url)
    mainhtml = req.text
    #print(len(mainhtml))
    soup = BeautifulSoup(mainhtml, "lxml")
    mainlink_region = soup.findAll("section", {"id" : "support-topics"}) #{"class":"side_tabs nav nav-tabs"})
    #print(len(mainlink_region))
    links = []
    if len(mainlink_region) > 0:
        linkregion = mainlink_region[0]
        
        #boxes = linkregion.findAll("h2")
        boxes = linkregion.findAll("div", {"class" : "box"})
        for i,box in enumerate(boxes):
            #print(i,box)
            #print(i)
            headerregion = box.select("h2")[0]
            header = headerregion.select("a")[0].text
            header_link = headerregion.select("a")[0]["href"]
            
            title = {}
            title["title"] = header
            title["link"] = head + header_link
            
            
            sublinkregion = box.findAll("ul", {"class" : "list-carrots"})[0]
            sublinks = sublinkregion.findAll("a")
            subtitles = []
            for sublink in sublinks:
                #pages[header_link] = pages[header_link] + [sublink.text]
                subtitles.append((sublink.text, head + sublink["href"]))
                #print(" ", sublink.text)
            #print()
            title["sublinks"] = subtitles
            pages.append(title)
            #title = header.select("a").text
            #title_link = header.select("a")[0]["href"]
            #print(title)
            #print(header)
            '''
            title = box.select("a")[0].text
            title_link = box.select("a")[0]["href"]
            pages[title_link] = []
            
            sublinkregion = linkregion.findAll("ul", {"class" : "list-carrots"})
            print("----",len(sublinkregion),"____")
            '''
            
            '''
            sublinks = sublinkregion.findAll("a")
            for sublink in sublinks:
                pages[title_link] = pages[title_link] + [sublink.text]
            print()
            ''' 
        
        #links = [a.text for a in link_text.findAll("a")]
    
    #print("********\n", pages)
    return pages



def collect_sublinks(url):
    
    req = requests.get(url)
    html = req.text

    soup = BeautifulSoup(html, "lxml")
    link_region = soup.findAll("ul", {"id" : "tabs-questions"}) #{"class":"side_tabs nav nav-tabs"})

    links = []
    if len(link_region) > 0:
        link_text = link_region[0]
        #links = [a.text for a in link_text.findAll("a")]
        for a in link_text.findAll("a"):
            links.append({"title" : a.text, "link" : a["href"]})
    
    return links
    

def _collectsublinks():
    
    
    root = "http://www.turkcell.com.tr/yardim/hattiniz"
    subpages = ["faturali", "hazirkart", "abonelik-islemleri", "sikca-sorulan-sorular"]
    
    urls = [root + "/" + s for s in subpages]


    for url in urls:
        
        print(url)
        req = requests.get(url)
        html = req.text
        print(len(html))
        soup = BeautifulSoup(html, "lxml")
        link_region = soup.findAll("ul", {"id" : "tabs-questions"}) #{"class":"side_tabs nav nav-tabs"})
        print(len(link_region))
        links = []
        if len(link_region) > 0:
            link_text = link_region[0]
            links = [a.text for a in link_text.findAll("a")]
        
        print(links)



if __name__ == '__main__':

    
    #pages = collect_all_links()
    
    #print(pages[0].keys())
    
    import pickle
    pages = pickle.load(open("/home/dicle/Documents/experiments/tcell_topics/pages.b", "rb"))
    
    l = collect_sublinks("http://www.turkcell.com.tr/yardim/turkcell-akademi/sikca-sorulan-sorular/turkcell-akademi-nedir")
    for i in l:
        print(" "*4 + i["title"])
    
    '''
    # subtitles in docs; only headers as docs
    docs = []
    for page1 in pages:
        
        title1 = page1["title"].strip().replace("/", "")
        link2 = page1["link"]
        links2 = page1["sublinks"]
        
        print(title1)
        
        doc_title = "-".join(title1.split())
        sentences = []
        
        for page2 in links2:
            
            title2 = page2["title"].strip().replace("/", "")
            link2 = page2["link"]
            links3 = page2["sublinks"]
            
            print(title2)    
            sentences.append(title2)
            
            for page3 in links3:
                
                title3 = page3["title"].strip()
                link3 = page3["link"]
                
                print("    ", title3)
                
                sentences.append(" "*4 + title3)
            
        docs.append({"doc_title" : doc_title,
                     "text" : "\n".join(sentences)})
        print()
        
    import os    
    outfolder = "/home/dicle/Documents/experiments/tcell_topics/docs_less"
    for doc in docs:
        title = doc["doc_title"]
        fpath = os.path.join(outfolder, title+".txt")
        print(title)
        print(fpath)
        with open(fpath, "w") as f:
            f.write(doc["text"])
    
    '''    
    
    
    '''
    # sub titles as docs
    docs = []
    for page1 in pages:
        
        title1 = page1["title"].strip()
        link2 = page1["link"]
        links2 = page1["sublinks"]
        
        print(title1)
        
        for page2 in links2:
            
            title2 = page2["title"].strip()
            link2 = page2["link"]
            links3 = page2["sublinks"]
            
            print("  ", title2)    
            
            doc_title = "-".join(title1.split()) + "_" + "-".join(title2.split())
            doc_title = doc_title.replace("/", "")
            sentences = []
            
            for page3 in links3:
                
                title3 = page3["title"].strip()
                link3 = page3["link"]
                
                print("    ", title3)
                
                sentences.append(title3)
            docs.append({"doc_title" : doc_title,
                         "text" : "\n".join(sentences)})
        print()
        
    import os    
    outfolder = "/home/dicle/Documents/experiments/tcell_topics/docs"
    for doc in docs:
        title = doc["doc_title"]
        fpath = os.path.join(outfolder, title+".txt")
        print(title)
        print(fpath)
        with open(fpath, "w") as f:
            f.write(doc["text"])
    '''
        