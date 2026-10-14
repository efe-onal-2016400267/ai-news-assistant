from typing import TypedDict


class QueryGroup(TypedDict):
    QueryType: str
    Queries: list[str]
    GoldLabels: list[list[str]]


class QueryConfs(TypedDict):
    config_ids: list[str]
    queries: list[QueryGroup]


query_confs: QueryConfs = {
    "config_ids": ["words-tok256-32", "recursive-tok256-32"],
    "queries": [
        {
            "QueryType": "Natural Language",
            "Queries": [
                "What did Mistral announce about their new Large model?",
                "On what devices can I use Muse?",
                "What incidents have involved AI agents acting outside their authorized limits?",
            ],
            "GoldLabels": [
                [
                    "183fcacc2014829eef79993ddc4b383d78b4a61bf99fd779560c7cb8b4184b7c",
                    "19bbd8a9667ab1144b6d4c57aa71c6c22c888724baa2465b03c7ecc12f76f962",
                    "d9a7e52b4293657c560cf29d1e330dbc21352cdd1e1eda7bd0a75fbce4703c0b",
                    "1ef243007176e85cc8c61f991c732cb67f59da2c4bf9bd4e693a0d6d5af2a173",
                    "0e8219bd64331ee357c25f4556db1efc67f48c65486ccb0d0e6227a9561b74c7",
                ],
                [
                    "552629978e356fab438cabde0c8e53f46c3fe97752b7e0b1108280817ac43f63",
                    "f213b8d3eadf23cd6b0caaf2a2add87658fe10bc75cb0864a2de6470d7624f70",
                    "5f645da35744d30c39db452f72fd559b26610176ac97f80a98ea6b30f6493264",
                ],
                [
                    "a43c2e0816bbaf3df57d7c1eab396d04ca37d071f2a883a46e814c48711647c3",
                    "00521fe0b8a575645038625a3fb32ba612d40710b59c6566975842203615df28",
                    "a919ba35a5e056c2278ba024862c09052a8b0e23df4c4932675faf972b48ed1b",
                    "3615960163537c76b3947126a9cb5765341bbc41b70ca78e0d3a995ca9b33af1",
                    "0d8c74eec04dc0baad67974a38e9d7e01fea672383ac18bd5e7828dc360726d0",
                    "1d64cace9beff2adfe9e2ce2038ee7fbbfe2413158594a57555685f89122c464",
                    "8088c1dfd69d52169d26c3ad98d416ee06b8c8b567835bffc63d0babec849503",
                    "e5230d9350e4e50fd2099f6ff9c43d57e3855bc97e4b51456db9a57a8808eb4f",
                ],
            ],
        },
        {
            "QueryType": "Exact Term",
            "Queries": [
                "What is Leanstral?",
                "What changed in Claude Haiku 5.5?",
                "What is celld?",
            ],
            "GoldLabels": [
                [
                    "1929444a4216300742aa7054989f6fe7c4c6224c580b4d8746c871fbff87114c",
                    "2baf5368ef46a143cf5b22f3730fc5f5f507984f0ef010d5b85e1bb50f33109b",
                ],
                [
                    "e1a7b06651e9c94175b39b61132cdb1e50059ee7794a525ad64130092b851b69",
                    "231befe88d753d0247c6f00f2b741e2a9ffdb254a9681c1fb308792afc321c01",
                ],
                [
                    "1d5fd2024caedd1b400985111d3b3bb167cf7d3afc72522081dee57fdf9f820c",
                ],
            ],
        },
        {
            "QueryType": "Paraphrase",
            "Queries": [
                "How big is the French lab’s new flagship model?",
                "Does letting a small local model think step by step fix its large-number"
                " arithmetic?",
                "How can I avoid a surprise cloud bill from a runaway agent?",
            ],
            "GoldLabels": [
                [
                    "183fcacc2014829eef79993ddc4b383d78b4a61bf99fd779560c7cb8b4184b7c",
                    "19bbd8a9667ab1144b6d4c57aa71c6c22c888724baa2465b03c7ecc12f76f962",
                    "d9a7e52b4293657c560cf29d1e330dbc21352cdd1e1eda7bd0a75fbce4703c0b",
                    "1ef243007176e85cc8c61f991c732cb67f59da2c4bf9bd4e693a0d6d5af2a173",
                    "0e8219bd64331ee357c25f4556db1efc67f48c65486ccb0d0e6227a9561b74c7",
                ],
                [
                    "69f5bd9058f87f8c2c946d7c872bc19a4da5cc270c76b24e9a59484cc7c89f63",
                ],
                [
                    "345f1755705d282a19e9b6b2dfd3b058484aab81833d520923e34990833223b2",
                ],
            ],
        },
        {
            "QueryType": "SeveralChunks",
            "Queries": [
                "What did OpenAI announce for security work at DevDay?",
                "In which areas is Mistral Large 4 strongest?",
            ],
            "GoldLabels": [
                [
                    "d3b185d971abc0ead80775d6b522b2f0118643f19e66b6283a753505d11aa945",
                ],
                [
                    "1ef243007176e85cc8c61f991c732cb67f59da2c4bf9bd4e693a0d6d5af2a173",
                ],
            ],
        },
        {
            "QueryType": "TimeSensitive",
            "Queries": [
                "Is Claude Haiku 5.5 available yet?",
                "What is the newest version of Leanstral?",
            ],
            "GoldLabels": [
                [
                    "e1a7b06651e9c94175b39b61132cdb1e50059ee7794a525ad64130092b851b69",
                    "231befe88d753d0247c6f00f2b741e2a9ffdb254a9681c1fb308792afc321c01",
                ],
                [
                    "1929444a4216300742aa7054989f6fe7c4c6224c580b4d8746c871fbff87114c",
                ],
            ],
        },
    ],
}
