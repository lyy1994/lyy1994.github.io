---
layout: post
title:  "利用Github pages+jekyll搭建个人博客"
date:   2016-03-25 17:25:55 +0800
categories: jekyll
---
1.[Github Pages官方指南] 建好Github仓库；

2.[两行命令部署jekyll] 部署jekyll；

	如果gem install jekyll出现：
	ERROR:  While executing gem ... (Gem::RemoteFetcher::FetchError)
	        Errno::ECONNRESET: An existing connection was forcibly closed by the remote host. - SSL_connect (https://api.rubygems.org/quick/Marshal.4.8/jekyll-3.1.2.gemspec.rz)
	那么尝试gem sources -a http://rubygems.org/ 然后再次输入gem install jekyll；
	
3.安装好jekyll就在命令行里面输入jekyll new myblog，然后把myblog里面的内容复制到username.github.io这个本地仓库里面，接着命令行路径转到这个仓库下，输入jekyll serve，然后在浏览器里面输入localhost:4000即可进行本地预览；

4.按照个人喜好修改好_config.yml后就可以直接使用github desktop提交changes并且sync，稍等一会在浏览器里面键入username.github.io就可以看到自己的博客了；

5.更多：[Jekyll官网]

注：

*我的系统Windows7，32位，git版本Github desktop。

*发表新博客最好从_post里面自带的post复制一份使用Notepad++修改出来，这样可以避免一些编码问题。

**---------------------------------------------2017/6/15 UPDATE---------------------------------------------**

在Windows7, 32位系统中，安装Jekyll只需要两步:

1. 在[Ruby官网]上选择安装最新版本的Ruby，并打开命令行输入如下命令检测是否成功安装Ruby和Gym：

```
ruby -v
gem -v
```

2. 打开命令行输入如下命令即可成功安装Jekyll：

```
gem install jekyll
```


[Github Pages官方指南]: https://pages.github.com/
[两行命令部署jekyll]: https://davidburela.wordpress.com/2015/11/28/easily-install-jekyll-on-windows-with-3-command-prompt-entries-and-chocolatey/
[Jekyll官网]: https://jekyllrb.com/docs/home/
[Ruby官网]: https://rubyinstaller.org/downloads/
