'''
Created on May 2015

@author: Arthur Glowacki, Argonne National Laboratory

Copyright (c) 2013, Stefan Vogt, Argonne National Laboratory
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

    Redistributions of source code must retain the above copyright notice, this
        list of conditions and the following disclaimer.
    Redistributions in binary form must reproduce the above copyright notice, this
        list of conditions and the following disclaimer in the documentation and/or
        other materials provided with the distribution.
    Neither the name of the Argonne National Laboratory nor the names of its
    contributors may be used to endorse or promote products derived from this
    software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
SUCH DAMAGE.
'''

import smtplib
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart

class mailman:
	def __init__(self, smtp_address, from_address, username, password):
		self.username = username
		self.password = password
		self.from_address = from_address
		self.smtp_address = smtp_address

	def send(self, to_addresses, subject, message, images_dict=None):
		if images_dict == None:
			msg = MIMEText(message)
		else:
			msg = MIMEMultipart()
			msg_txt = MIMEText(message)
			msg.attach(msg_txt)
		msg['Subject'] = subject
		msg['From'] = self.from_address
		msg['To'] = to_addresses
		if images_dict != None:
			img_text = '<br><br>'
			for img_name in images_dict.iterkeys():
				msg_img = MIMEImage(images_dict[img_name], Name=img_name)
				msg_img.add_header('Content-ID', '<' + img_name + '>')
				msg.attach(msg_img)
				img_text += '<b>' + img_name + '</b><br><img src="cid:' + img_name + '"><br><br>'
			msgText = MIMEText(img_text, 'html')
			msg.attach(msgText)
		smtp = smtplib.SMTP(self.smtp_address)
		smtp.ehlo()
		smtp.starttls()
		smtp.login(self.username, self.password)
		smtp.sendmail(self.from_address, to_addresses, msg.as_string())
		smtp.quit()
