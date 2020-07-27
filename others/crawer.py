# coding=utf-8
"Selenium是一个自动化测试工具，利用它可以驱动浏览器执行特定的动作，如点击、下拉等操作"
"同时还可以获取浏览器当前呈现的页面的源代码，做到可见即可爬。对于一些JavaScript动态渲染的页面来说，这种抓取方式十分有效。"
from selenium import webdriver
import os
from lxml import etree
import time
from wordcloud import WordCloud
import matplotlib.pyplot as plt  # 绘制图像的模块
import jieba  # jieba分词

#生成词云
def get_wordcloud(file):
    f = open(file, 'r', encoding='gbk').read()

    # 结巴分词，生成字符串，wordcloud无法直接生成正确的中文词云
    cut_text = " ".join(jieba.cut(f))

    wordcloud = WordCloud(
        # 设置字体，不然会出现口字乱码，文字的路径是电脑的字体一般路径，可以换成别的
        font_path="C:/Windows/Fonts/simfang.ttf",
        # 设置了背景，宽高
        background_color="white", width=2000, height=1380).generate(cut_text)

    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()


#获取QQ好友的说说内容(需模拟浏览器操作)
def get_qzone(qq):
    file = 'E:/{}.txt'.format(qq)
    driver = webdriver.Firefox()
    driver.maximize_window()  # 窗口最大化

    driver.get('https://user.qzone.qq.com/{}/311'.format(qq))  # URL
    driver.implicitly_wait(10)  # 隐示等待，为了等待充分加载好网址
    driver.find_element_by_id('login_div')
    driver.switch_to_frame('login_frame')  # 切到输入账号密码的frame
    driver.find_element_by_id('switcher_plogin').click()  ##点击‘账号密码登录’
    driver.find_element_by_id('u').clear()  ##清空账号栏
    driver.find_element_by_id('u').send_keys('292759322')  # 输入账号
    driver.find_element_by_id('p').clear()  # 清空密码栏
    driver.find_element_by_id('p').send_keys('rensheng/chujian')  # 输入密码
    driver.find_element_by_id('login_button').click()  # 点击‘登录’
    driver.switch_to_default_content()  # 跳出当前的frame，这步很关键，不写会报错的，因为你登录后还要切刀另一个frame

    driver.implicitly_wait(10)
    time.sleep(3)

    try:
        driver.find_element_by_id('QM_OwnerInfo_Icon')  # 判断是否QQ空间加了权限
        b = True
    except:
        b = False

    if b == True:
        page = 1;

        try:
            while page:
                ##下拉
                for j in range(1, 5):
                    driver.execute_script("window.scrollBy(0,5000)")
                    time.sleep(2)

                driver.switch_to_frame('app_canvas_frame')  # 切入说说frame
                selector = etree.HTML(driver.page_source)
                title = selector.xpath('//li/div/div/pre/text()')  ##获取title集合
                print(title)

                for i in title:
                    if not os.path.exists(file):
                        print('创建TXT成功')

                    with open(file, 'a+') as f:
                        f.write(i + '\n\n')
                        f.close()

                page = page + 1
                driver.find_element_by_link_text(u'下一页').click()  # 点击下一页
                driver.switch_to.default_content()  # 跳出当前frame
                time.sleep(3)
            driver.quit()
        except Exception as e:
            # 我没有判断什么时候为最后一页，当爬取到最后一页，
            # 默认点击下一页，会出现异常，我直接在这认为它是爬到末尾了，还是有待优化
            print("爬取完成，爬到的最后页数为" + str(page - 1))
            get_wordcloud(file)
            driver.quit()
            driver.close()


if __name__ == '__main__':
    get_qzone('1136152343')