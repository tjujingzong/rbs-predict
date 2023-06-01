# concatenation
class DeepEnsembleNet(nn.Module):
    def __init__(self, dl_list=None, num_classes=10):
        super(DeepEnsembleNet, self).__init__()
        # print('ok')
        self.num_classes = num_classes

        self.features = nn.ModuleList()

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential()

        self.init_layers(dl_list, num_classes)

    def init_layers_with_pretrained(self, m_list=None, is_requires_grad=True):
        if m_list is None:
            return

        out_features = []

        for m in m_list:
            cname = m.__class__.__name__
            out_feature = model_out_features_map[cname]
            feature = FeatureExtractor(m, is_requires_grad=is_requires_grad)
            self.features.append(feature)
            out_features.append(out_feature)

        last_channel = sum(out_features)

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            # nn.Linear(last_channel, int(last_channel/2)),
            # nn.Dropout(0.2),
            # nn.Linear(int(last_channel/2), int(last_channel/4)),
            # nn.Dropout(0.2),
            nn.Linear(last_channel, self.num_classes),
        )

    def init_layers(self, dl_list=None, is_requires_grad=True):

        if dl_list is None:
            return

        out_features = []

        for m in dl_list:
            model = load_model_from_lib(m)
            out_feature = model_out_features_map[m]
            feature = FeatureExtractor(model, is_requires_grad=is_requires_grad)
            self.features.append(feature)
            out_features.append(out_feature)

        last_channel = int(sum(out_features))

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            # nn.Linear(last_channel, int(last_channel/2)),
            # nn.Dropout(0.2),
            # nn.Linear(int(last_channel/2), int(last_channel/4)),
            # nn.Dropout(0.2),
            nn.Linear(last_channel, self.num_classes),
        )

    def features_forward(self, x):
        z = None
        for f in self.features:
            # print(f)
            # print(x.shape)
            y = f(x)
            # print(y.shape)
            y = self.pool(y)
            # print(y.shape)
            y = torch.flatten(y, 1)
            # output=output.view(output.size(0), -1)
            # print(y.shape)
            # z.append(y)
            # print(y.shape)
            if z is None:
                z = y
            else:
                z = torch.cat((z, y), dim=1)
        return z

    def forward(self, x):

        x = self.features_forward(x)
        # print(x.shape)
        # x=self.pool(x)
        x = x.view(x.size(0), -1)
        # print(x.shape)

        x = self.classifier(x)
        # print('ok')
        return x

    # attention


class DeepEnsembleNetV2(nn.Module):
    '''
    @author: hys
    @time  : 2020.10.22
    '''

    def __init__(self, dl_list=None, num_classes=10, feature_dim=256):
        super(DeepEnsembleNetV2, self).__init__()
        self.num_classes = num_classes
        self.feature_extractors = nn.ModuleList()
        self.necks = nn.ModuleList()
        self.classifiers = nn.ModuleList()
        self.feature_dim = feature_dim
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.init_layers(dl_list, num_classes)

    def init_layers_with_pretrained(self, m_list=None, is_requires_grad=True, dataid=4):
        if m_list is None:
            return

        for m in m_list:
            out_feature = model_out_features_map[m]
            model = load_model_from_lib(m)
            checkpoint = torch.load(os.path.join(config.checkpoint_path, f'{m}_{dataid}_228.pt'), map_location='cuda')[
                'state_dict']
            model.load_state_dict(checkpoint)
            feature = FeatureExtractor(model, is_requires_grad=is_requires_grad)
            self.feature_extractors.append(feature)
            self.necks.append(
                nn.Sequential(
                    nn.Dropout(0.2),
                    nn.Linear(out_feature, self.feature_dim)
                )
            )
            self.classifiers.append(
                nn.Sequential(
                    nn.Dropout(0.2),
                    nn.Linear(self.feature_dim, self.num_classes)
                )
            )

    def init_layers(self, dl_list=None, is_requires_grad=True):

        if dl_list is None:
            return

        for m in dl_list:
            model = load_model_from_lib(m)
            out_feature = model_out_features_map[m]
            feature = FeatureExtractor(model, is_requires_grad=is_requires_grad)
            self.feature_extractors.append(feature)
            self.necks.append(
                nn.Sequential(
                    nn.Dropout(0.2),
                    nn.Linear(out_feature, self.feature_dim)
                )
            )
            self.classifiers.append(
                nn.Sequential(
                    nn.Dropout(0.2),
                    nn.Linear(self.feature_dim, self.num_classes)
                )
            )

    def features_forward(self, x):
        features = []
        for f in self.feature_extractors:
            y = f(x)
            # print(x.shape, y.shape)
            y = self.pool(y)
            # print(y.shape)
            y = torch.flatten(y, 1)
            # print(y.shape)
            y = torch.unsqueeze(y, 1)
            # print(y.shape)
            features.append(y)
        return features

    def dim_specification(self, features):
        features_with_same_dim = []
        for i, f in enumerate(features):
            features_with_same_dim.append(self.necks[i](f))
        return features_with_same_dim

    def attention(self, query, key, value):
        # query dim: (batch, 1, feature_dim)
        # key and value dim: (batch, m, feature_dim) and m is the number of feature_extractors
        weights = torch.bmm(query, torch.transpose(key, 1, 2))
        weights = F.softmax(weights, dim=2)
        # print(weights)
        return torch.bmm(weights, value)

    def classification(self, features_with_same_dim):
        # print(features_with_same_dim[0].shape)
        res = []
        key = torch.cat(features_with_same_dim, dim=1)
        value = torch.cat(features_with_same_dim, dim=1)
        # print(key.shape, value.shape)
        for i, f in enumerate(features_with_same_dim):
            atten = self.attention(f, key, value)
            res.append(torch.squeeze(self.classifiers[i](atten), 1))
        return res

    def forward(self, x):
        # print(x.shape)
        features = self.features_forward(x)
        features_with_same_dim = self.dim_specification(features)
        res = self.classification(features_with_same_dim)
        # print(res[0].shape)
        res = torch.cat([torch.unsqueeze(item, dim=1) for item in res], dim=1)
        res = torch.mean(res, dim=1, keepdims=False)
        # print(res.shape)
        return res


# 
class DeepEffEnsembleNet(nn.Module):
    def __init__(self, dl_list=None, num_classes=10, out_channels=1024):
        super(DeepEffEnsembleNet, self).__init__()
        # print('ok')
        self.num_classes = num_classes

        self.features = nn.ModuleList()

        self.out_channels = out_channels

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential()

        self.init_layers(dl_list, num_classes)

    def init_layers_with_pretrained(self, m_list=None, is_requires_grad=True):
        if m_list is None:
            return

        out_features = []

        for m in m_list:
            cname = m.__class__.__name__
            out_feature = model_out_features_map[cname]
            feature = FeatureExtractor(m, is_requires_grad=is_requires_grad)
            self.features.append(feature)
            out_features.append(out_feature)

        last_channel = sum(out_features)

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            # nn.Linear(last_channel, int(last_channel/2)),
            # nn.Dropout(0.2),
            # nn.Linear(int(last_channel/2), int(last_channel/4)),
            # nn.Dropout(0.2),
            nn.Linear(last_channel, self.num_classes),
        )

    def init_layers(self, dl_list=None, is_requires_grad=True):

        if dl_list is None:
            return

        out_features = []

        for m in dl_list:
            model = load_model_from_lib(m)
            out_feature = model_out_features_map[m]
            feature = FeatureExtractor(model, is_requires_grad=is_requires_grad)
            self.features.append(feature)
            out_features.append(out_feature)

        last_channel = int(sum(out_features))

        self.neck = nn.Conv2d(last_channel, self.out_channels, kernel_size=(1, 1))

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            # nn.Linear(last_channel, int(last_channel/2)),
            # nn.Dropout(0.2),
            # nn.Linear(int(last_channel/2), int(last_channel/4)),
            # nn.Dropout(0.2),
            nn.Linear(self.out_channels, self.num_classes),
        )

    def features_forward(self, x):
        z = None
        for f in self.features:
            y = f(x)
            if z is None:
                z = y
            else:
                z = torch.cat((z, y), dim=1)
        return z

    def forward(self, x):

        x = self.features_forward(x)
        # print(x.shape)
        x = self.neck(x)
        # print(x.shape)
        x = self.pool(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        # print(x.shape)

        x = self.classifier(x)
        # print('ok')
        return x
