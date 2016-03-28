---
layout: post
title:  "给LaTeX初学者一点建议"
date:   2016-03-26 16:40:45 +0800
categories: LaTeX
---
利用LaTeX进行写作，我认为有以下几个原则：

1.保持一致。让tex文件的代码看上去跟编译过后的dvi或者PDF一样。表格示例：

代码:

	\begin{tabular}{c r @{.} l}
		Pi expression       & \multicolumn{2}{c}{Value} \\
		\hline
		$\pi$               &           3&1416          \\
		$\pi^{\pi}$         &          36&46            \\
		$(\pi^{\pi})^{\pi}$ &       80662&7             \\
	\end{tabular}	
	
编译：

![](/images/LaTeX1-1.jpg)

2.保持稀疏。善于利用LaTeX对空格不敏感的特性，把代码弄稀疏一点，如使用缩进保持内容的整洁，使用空行划分文章不同部分（每段隔一个空行，每个section之间多隔几个空行）。

3.模块化。长的文章最好按照其主题分成不同文件再组合一起。

4.要善于使用\emph{}、\textbf{}等功能来强调重点，如果允许，多插入图片对文档进行说明，使文档更通俗易懂。

5.善用注释。对于长文档，不同部分要及时添加注释，否则容易遗忘其内容导致浪费时间去回顾代码。

6.自定义命令。对于经常需要插入相似的长文本或者公式的情况，自定义命令会让文档看起来更简洁，写得更轻松。

7.介绍两个可以在线编译LaTeX文档的网站，避免安装环境的麻烦而且即时编译使用更加方便：

[Overleaf]

[ShareLaTeX]

[Overleaf]: https://www.overleaf.com/
[ShareLaTeX]: https://www.sharelatex.com/
