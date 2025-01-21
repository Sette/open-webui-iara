

from typing import IO
import logging
import requests

import asyncio
import aiohttp
import aiofiles

logger = logging.getLogger(__name__)


class PdftotextLoader:
    def __init__(self, pdf_path: str, url: str, max_pages: int):
        # Ajusta a URL para incluir o endpoint específico
        url += "/api-ds-ocr/text_extract"
        self.url = url
        self.pdf_path = pdf_path
        self.max_pages = max_pages

    async def load(self):
        """
        Faz upload do PDF (via POST assíncrono) para o serviço de OCR 
        e retorna o texto extraído, ou None em caso de erro.
        """
        try:
            # Leitura assíncrona do arquivo PDF
            async with aiofiles.open(self.pdf_path, "rb") as f:
                pdf_content = await f.read()

            # Monta cabeçalhos
            headers = {
                "accept": "application/json",
                # Inclua outros cabeçalhos caso precise (ex: Authorization)
            }

            # Monta um FormData com o arquivo e os campos adicionais
            form = aiohttp.FormData()
            form.add_field(
                "pdf_upload",
                pdf_content,
                content_type="application/pdf"  # tipo de conteúdo
            )
            form.add_field("max_pages", self.max_pages)
            form.add_field("header_footer", False)

            # Cria a sessão de conexão e faz a requisição POST
            async with aiohttp.ClientSession() as session:
                # Ajuste o timeout conforme a necessidade (em segundos)
                async with session.post(self.url, data=form, headers=headers, timeout=600) as response:
                    if response.status == 200:
                        # Lê a resposta como JSON
                        data_json = await response.json()
                        # Extrai o campo "text"
                        txt = data_json.get("text", "")
                        logger.info(
                            "Extraído texto do PDF via OCR. Tamanho do texto: %d caracteres", 
                            len(txt)
                        )
                        return txt
                    else:
                        # Se não for 200, tente ler o body para logging/debug
                        error_text = await response.text()
                        logger.error(
                            "Erro ao enviar PDF para %s (status: %d). Detalhes: %s",
                            self.url, response.status, error_text
                        )
                        return None

        except asyncio.TimeoutError:
            logger.error(f"Timeout ao enviar dados para: {self.url}")
            return None
        except Exception as e:
            logger.exception(f"Erro ao enviar dados para {self.url}: {e}")
            return None